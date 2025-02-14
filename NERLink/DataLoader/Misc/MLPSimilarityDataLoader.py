#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    04/30/2022                                                                   #
#    Revised: 11/12/2022                                                                   #
#                                                                                          #
#    Concept Linking MLP Embedding Similarity Model -  Data Loader                         #
#       Loads Term-to-Concept Instance File In Addition To Term And Concept Embeddings.    #
#                                                                                          #
#    Term-to-Concept Instance File Format:                                                 #
#    -------------------------------------                                                 #
#       term_a\tconcept_a                                                                  #
#       term_b\tconcept_b                                                                  #
#             ...                                                                          #
#       term_n\tconcept_n                                                                  #
#                                                                                          #
#                                                                                          #
#    Embedding File Format: Standard Word2vec Plain Text                                   #
#                                                                                          #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import re
import numpy as np

# Custom Modules
from NERLink.DataLoader.Base import DataLoader


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class MLPSimilarityDataLoader( DataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  skip_individual_mentions = False, skip_composite_mentions = False, lowercase = False, ignore_label_type_list = [] ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle, lowercase = lowercase,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          skip_individual_mentions = skip_individual_mentions, skip_composite_mentions = skip_composite_mentions,
                          ignore_label_type_list = ignore_label_type_list )
        self.version             = 0.01
        self.max_sequence_length = 1

    """
        Reads Embedding Similarity Instances Data
    """
    def Read_Data( self, file_path, lowercase = True, keep_in_memory = True, encode_strings_to_utf8 = True ):
        # Check(s)
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "MLPSimilarityDataLoader::Read_Data() - Error: File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return self.data_list

        # Store File Path
        self.data_file_path = file_path

        # Store Lowercase Setting In DataLoader Parent Class (Used For self.Get_Token_ID() Function Calls)
        self.lowercase_text = lowercase

        data_list = self.utils.Read_Data( file_path = file_path, lowercase = lowercase )

        if keep_in_memory:
            self.Print_Log( "MLPSimilarityDataLoader::Read_Data() - Storing Processed Data In Memory" )
            self.data_list = data_list

        self.Print_Log( "MLPSimilarityDataLoader::Read_Data() - Complete" )

        return data_list

    """
        Encodes Concept Linking Data - Used For Training, Validation Or Evaluation Data

        Inputs:
            data_list          : List Of Passage Objects
            keep_in_memory     : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            is_validation_data : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
            concept_delimiter  : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                 Used For One-To-Many Relationships

        Outputs:
            concept_inputs     : Numpy Array
            concept_outputs    : Numpy Array
    """
    def Encode_CL_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = False, keep_in_memory = True,
                              is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                              mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = True,
                              use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Encoding Concept Instances" )

        encoded_term_inputs, encoded_concept_outputs = [], []

        # Encode All Term and Concept Pairs
        for term_concept_instance in data_list:
            term, concept = term_concept_instance.split( '\t' )

            # Check To See If Concept Is In Ignore List
            if concept in self.ignore_label_type_list:
                self.Print_Log( "MLPSimilarityDataLoader::Encode_ML_Model_Data() - Concept In Label Ignore List / Skipping Concept: \"" + str( concept ) + "\"" )
                continue

            encoded_term, encoded_concept = self.Encode_CL_Instance( entry_term = term, annotation_concept = concept )

            if encoded_term is None:
                self.Print_Log( "MLPSimilarityDataLoader::Encode_ML_Model_Data() - Term Encoding Error - Term:" + str( term ) + "\n" )
                continue
            if encoded_concept is None:
                self.Print_Log( "MLPSimilarityDataLoader::Encode_ML_Model_Data() - Concept Encoding Error - Concept:" + str( concept ) + "\n" )
                continue

            encoded_term_inputs.append( encoded_term )
            encoded_concept_outputs.append( encoded_concept )

        # Convert Data To Numpy Arrays
        encoded_term_inputs     = np.asarray( encoded_term_inputs,     dtype = np.float32 )
        encoded_concept_outputs = np.asarray( encoded_concept_outputs, dtype = np.float32 )

        # Check(s)
        number_of_input_instances  = encoded_term_inputs.shape[0]
        number_of_output_instances = encoded_concept_outputs.shape[0]

        if number_of_input_instances == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Inputs", force_print = True )
            return None, None
        elif number_of_output_instances == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Outputs", force_print = True )
            return None, None
        elif number_of_input_instances != number_of_output_instances:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Error: Number Of Input And Output Instances Not Equal", force_print = True )
            return None, None

        if keep_in_memory:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Storing Encoded Data In Memory" )

            if is_validation_data:
                self.concept_val_inputs   = encoded_term_inputs
                self.concept_val_outputs  = encoded_concept_outputs
            elif is_evaluation_data:
                self.concept_eval_inputs  = encoded_term_inputs
                self.concept_eval_outputs = encoded_concept_outputs
            else:
                self.concept_inputs       = encoded_term_inputs
                self.concept_outputs      = encoded_concept_outputs

        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Model_Data() - Complete" )

        return encoded_term_inputs, encoded_concept_outputs

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            entry_term         : Concept Token (String)
            annotation_concept : Concept Token MeSH ID / CUI (String)
            concept_delimiter  : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                 Used For One-To-Many Relationships
            pad_output         : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                 Categorical Crossentropy vs. Sparse Categorical Crossentropy

        Outputs:
            encoded_entry_term : Encoded Entry Term Vector
            encoded_concept    : Encoded Concept Vector (List/Vector Of Integers)
    """
    def Encode_CL_Instance( self, entry_term, annotation_concept, text_sequence = None, annotation_indices = None, pad_input = True, pad_output = False,
                            concept_delimiter = None, mask_term_sequence = False, separate_sentences = True, term_sequence_only = False,
                            restrict_context = False, label_per_sub_word = False ):
        # Check
        if len( self.concept_id_dictionary ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None, None

        encoded_term, encoded_concept = [], []

        # Map Entry Terms To Concepts
        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() - Encoding Entry Term/Concept To Concept ID" )
        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() -      Entry Term: " + str( entry_term         ) )
        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() -      Concept   : " + str( annotation_concept ) )

        # Prioritize Compoundified Term
        compoundifed_term = re.sub( r'\s+', "_", entry_term )

        if compoundifed_term in self.token_id_dictionary:
            encoded_term = self.Get_Embeddings_A()[self.Get_Token_ID( compoundifed_term )]
        # Encode Entry Term / Term Exists In Dictionary
        elif entry_term in self.token_id_dictionary:
            encoded_term = self.Get_Embeddings_A()[self.Get_Token_ID( entry_term )]
        # Try To Compute Average Representation Of Term
        #   NOTE: Only Computes Average Of Terms Found
        else:
            for word in entry_term.split():
                if word in self.token_id_dictionary:
                    encoded_term.append( self.embeddings_a[self.Get_Token_ID( word )] )

            if len( encoded_term ) > 1:
                encoded_term = sum( encoded_term ) / len( encoded_term )
            elif len( encoded_term ) == 1:
                encoded_term = encoded_term[0]

        # Encode Concept
        if annotation_concept in self.concept_id_dictionary:
            encoded_concept = self.Get_Embeddings_B()[self.Get_Concept_ID( annotation_concept )]

        # Check(s)
        if len( encoded_term ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() - Error Encoding Term - Term: " + str( entry_term ), force_print = True )
            return None, None

        if len( encoded_concept ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() - Error Encoding Concept - Concept: " + str( annotation_concept ), force_print = True )
            return None, None

        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() -      Encoded Entry Term: " + str( encoded_term   ) )
        self.Print_Log( "MLPSimilarityDataLoader::Encode_CL_Instance() -      Encoded Concept   : " + str( encoded_concept ) )

        return encoded_term, encoded_concept

    """
        Decodes Input Sequence Instance Of IDs To Entry Term String(s)

        Note: This Assume Only One Entry Term Exists Per Input Sequence.

        Inputs:
            encoded_input_instance : List/Numpy Array Containing Term Embedding (List/Numpy Array)

        Outputs:
            decoded_entry_term     : Decoded Entry Term (String)
    """
    def Decode_CL_Input_Instance( self, encoded_input_instance, entry_term_mask = None ):
        # Check(s)
        if isinstance( encoded_input_instance, list ) or isinstance( encoded_input_instance, np.ndarray ) \
           and len( encoded_input_instance ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Input_Instance() - Error: Encoded Term Instance List Is Empty" )
            return ""

        if isinstance( encoded_input_instance, np.ndarray ):
            encoded_input_instance = np.asarray( encoded_input_instance, dtype = np.float32 )

        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Input_Instance() - Term ID: " + str( encoded_input_instance ) )

        decoded_term, decoded_term_id = None, -1

        for idx, embedding in enumerate( self.Get_Embeddings_A() ):
            embedding = np.asarray( embedding, dtype = np.float32 )
            if embedding == encoded_input_instance:
                decoded_term_id = idx
                decoded_term    = self.Get_Token_From_ID( idx )

        if not decoded_term: decoded_term = "N/A"

        if decoded_term_id == -1:
            self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Input_Instance() - Error: Term Embedding Does Not Exist In Token Dictionary / Maybe Average Among Many Embeddings" )
            return ""

        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Input_Instance() - Decoding Output Label Instance: " + str( decoded_term ) )

        return decoded_term

    """
        Decodes Output Instance Of Labels For Concept Linking To List Of Concept ID Strings

        Inputs:
            encoded_output_labels : Model Prediction Output Labels Values (List/Numpy Array)

        Outputs:
            decoded_output_labels : Decoded Concept IDs (List)
    """
    def Decode_CL_Output_Instance( self, encoded_output_labels ):
        # Check(s)
        if isinstance( encoded_output_labels, list ) and len( encoded_output_labels ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_labels, np.ndarray ) and encoded_output_labels.shape[0] == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Output_Instance() - Decoding Output Embedding: " + str( encoded_output_labels ) )

        decoded_output_labels = None

        for idx, embedding in enumerate( self.Get_Embeddings_B() ):
            embedding = np.asarray( embedding, dtype = np.float32 )
            if embedding == encoded_output_labels:
                decoded_output_labels = self.Get_Concept_From_ID( idx )

        if not decoded_output_labels: decoded_output_labels = "N/A"

        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Output_Instance() - Decoded Output Label Instance: " + str( decoded_output_labels ) )

        return decoded_output_labels

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self, encoded_input_instance, entry_term_mask = None, encoded_output_labels = [] ):
        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Instance() - Encoded Sequence     : " + str( encoded_input_instance ) )
        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Instance() - Encoded Output Labels: " + str( encoded_output_labels ) )

        decoded_entry_term    = self.Decode_CL_Input_Instance( encoded_input_instance = encoded_input_instance )
        decoded_output_labels = self.Decode_CL_Output_Instance( encoded_output_labels = encoded_output_labels )

        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Instance() - Decoded Entry Term   : " + str( decoded_entry_term ) )
        self.Print_Log( "MLPSimilarityDataLoader::Decode_CL_Instance() - Decoded Output Labels: " + str( decoded_output_labels ) )

        return decoded_entry_term, decoded_output_labels


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    def Load_Token_ID_Key_Data( self, file_path ):
        if self.Get_Number_Of_Unique_Tokens() > 0 or self.Get_Number_Of_Unique_Concepts() > 0:
            self.Print_Log( "MLPSimilarityDataLoader::Load_Token_ID_Key_Data() - Warning: Primary Key Hash Is Not Empty / Saving Existing Data To: \"temp_primary_key_data.txt\"", force_print = True )
            self.Save_Token_ID_Key_Data( "temp_key_data.txt" )

        self.token_id_dictionary, self.concept_id_dictionary = {}, {}

        file_data = self.utils.Read_Data( file_path = file_path )

        if len( file_data ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Load_Token_ID_Key_Data() - Error Loading File Data: \"" + str( file_path ) + "\"" )
            return False

        self.Print_Log( "MLPSimilarityDataLoader::Load_Token_ID_Data() - Loading Key Data" )

        token_flag, concept_flag = False, False

        for idx in range( len( file_data ) ):
            if "<*> TOKEN ID DICTIONARY <*>" in file_data[idx]:
                token_flag   = True
                concept_flag = False
                continue
            if "<*> END TOKEN ID DICTIONARY <*>" in file_data[idx]:
                token_flag   = False
                concept_flag = False
                continue
            if "<*> CONCEPT ID DICTIONARY <*>" in file_data[idx]:
                token_flag   = False
                concept_flag = True
                continue
            if "<*> END CONCEPT ID DICTIONARY <*>" in file_data[idx]:
                token_flag   = False
                concept_flag = False
                continue

            key, value = file_data[idx].split( "<:>" )
            self.Print_Log( "MLPSimilarityDataLoader::Load_Token_ID_Data() - Key: " + str( key ) + " - Value: " + str( value ) )
            if token_flag:   self.token_id_dictionary[key]   = int( value )
            if concept_flag: self.concept_id_dictionary[key] = int( value )

        self.Print_Log( "MLPSimilarityDataLoader::Load_Token_ID_Data() - Complete" )

        return True

    def Save_Token_ID_Key_Data( self, file_path ):
        if len( self.token_id_dictionary ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Save_Token_ID_Key_Data() - Warning: Primary Key Data = Empty / No Data To Save" )
            return

        self.Print_Log( "MLPSimilarityDataLoader::Save_Token_ID_Data() - Saving Key Data" )

        # Open File Handle
        fh = open( file_path, "w", encoding = "utf8" )

        # Save Token ID Dictionary
        fh.write( "<*> TOKEN ID DICTIONARY <*>\n" )

        for key in self.token_id_dictionary:
            fh.write( str( key ) + "<:>" + str( self.token_id_dictionary[key] ) + "\n" )

        fh.write( "<*> END TOKEN ID DICTIONARY <*>\n" )

        # Save Concept ID Dictionary
        fh.write( "<*> CONCEPT ID DICTIONARY <*>\n" )

        for key in self.concept_id_dictionary:
            fh.write( str( key ) + "<:>" + str( self.concept_id_dictionary[key] ) + "\n" )

        fh.write( "<*> END CONCEPT ID DICTIONARY <*>\n" )

        fh.close()

        self.Print_Log( "MLPSimilarityDataLoader::Save_Token_ID_Data() - Complete" )

    """
        Generates IDs For Each Token Given The Following File Format

            Expected Format:    Seqences Of Text

        Inputs:
            data_list                    : List Of Sequences
            lowercase                    : True = Cased Text, False = Uncased Text (Bool)
            scale_embedding_weight_value : Scales Embedding Weights By Specified Value ie. embedding_weights *= scale_embedding_weight_value (Float)
            update_dict                  : Forces Function To Run, Updating The Token ID Dictionary (If Not Using Embeddings) (Bool)

        Outputs:
            None
    """
    def Generate_Token_IDs( self, data_list = [], lowercase = False, scale_embedding_weight_value = 1.0, update_dict = False, concept_delimiter = "," ):
        # Check(s)
        if self.generated_embedding_ids and update_dict == False:
            self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Warning: Already Generated Embedding Token IDs" )
            return

        if len( data_list ) > 0 and len( self.embeddings_a ) > 0 and self.generated_embedding_ids == False:
            self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Warning: Token IDs Cannot Be Generated From Data List When Embeddings Have Been Loaded In Memory" )
            return

        if update_dict:
            self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Updating Token ID Dictionary" )

        # Check(s)
        # If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        generated_embedding_ids = False

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        # Only Generate Token ID Dictionary Using Embeddings Once.
        #   This Is Skipped During Subsequent Calls To This Function
        if not self.generated_embedding_ids:
            # Generate Embeddings Based On Term Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_a ) > 0:
                # Insert Padding At First Index Of The Token ID Dictionary
                padding_token = self.padding_token.lower() if lowercase else self.padding_token
                self.token_id_dictionary[padding_token] = 0

                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_a ) + 1, len( self.embeddings_a[1].split() ) - 1 ) )

                self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Generating Token IDs Using Embeddings" )

                # Parse Embeddings
                for index, embedding in enumerate( self.embeddings_a, 1 ):
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
                        continue

                    # Tokenize Embedding String Into List
                    embedding_elements = embedding.split()

                    # Grab Number Of Embedding Text Elements
                    embedding_idx      = 1

                    # Count Number Of Embedding Elements Found Which Are Non-Numeric
                    for idx, embedding_value in enumerate( embedding_elements ):
                        if not re.search( r'^-*\d+\.\d+', embedding_value ): embedding_idx = idx + 1

                    # If None Are Found, Then Assume The First Element Is Our Embedding Text
                    #   i.e. First Element Is A Numerical Value And The Rest Its Embedding
                    if embedding_idx <= 0: embedding_idx = 1

                    # Join Embedding Text Elements Into Contiguous Text Span Separated By Spaces
                    embedding_text     = " ".join( embedding_elements[0:embedding_idx] )

                    if lowercase: embedding_text = embedding_text.lower()

                    # Slice Embedding Elements (Float Values)
                    embeddings[index]  = np.asarray( embedding_elements[embedding_idx:], dtype = 'float32' )

                    # Check To See If Element Is Already In Dictionary, If Not Add The Element
                    if embedding_text not in self.token_id_dictionary:
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( embedding_text ) + "\" => Embedding Row Index: " + str( index ) )
                        self.token_id_dictionary[embedding_text] = index
                    else:
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Adding Token - Warning: \"" + str( embedding_text ) + "\" Already In Dictionary" )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_input_tokens  = len( self.token_id_dictionary )

                self.embeddings_a = []
                self.embeddings_a = np.asarray( embeddings ) * scale_embedding_weight_value if scale_embedding_weight_value != 1.0 else np.asarray( embeddings )

                self.embeddings_a_loaded = True
                generated_embedding_ids  = True

            # Generate Embeddings Based On Concept Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_b ) > 0:
                # Insert CUI-Less At First Index Of The Token ID Dictionary
                cui_less_token = self.cui_less_token.lower() if lowercase else self.cui_less_token
                self.concept_id_dictionary[cui_less_token] = 0

                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_b ) + 1, len( self.embeddings_b[1].split() ) - 1 ) )

                self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Generating Token IDs Using Embeddings" )

                # Parse Embeddings
                for index, embedding in enumerate( self.embeddings_b, 1 ):
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
                        continue

                    # Tokenize Embedding String Into List
                    embedding_elements = embedding.split()

                    # Grab Number Of Embedding Text Elements
                    embedding_idx      = 1

                    # Count Number Of Embedding Elements Found Which Are Non-Numeric
                    for idx, embedding_value in enumerate( embedding_elements ):
                        if not re.search( r'^-*\d+\.\d+', embedding_value ): embedding_idx = idx + 1

                    # If None Are Found, Then Assume The First Element Is Our Embedding Text
                    #   i.e. First Element Is A Numerical Value And The Rest Its Embedding
                    if embedding_idx <= 0: embedding_idx = 1

                    # Join Embedding Text Elements Into Contiguous Text Span Separated By Spaces
                    embedding_text     = " ".join( embedding_elements[0:embedding_idx] )

                    if lowercase: embedding_text = embedding_text.lower()

                    # Slice Embedding Elements (Float Values)
                    embeddings[index]  = np.asarray( embedding_elements[embedding_idx:], dtype = 'float32' )

                    # Check To See If Element Is Already In Dictionary, If Not Add The Element
                    if embedding_text not in self.concept_id_dictionary:
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Adding Concept Token: \"" + str( embedding_text ) + "\" => Embedding Row Index: " + str( index ) )
                        self.concept_id_dictionary[embedding_text] = index
                    else:
                        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Adding Concept Token - Warning: \"" + str( embedding_text ) + "\" Already In Dictionary" )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_concept_tokens  = len( self.concept_id_dictionary )

                self.embeddings_b = []
                self.embeddings_b = np.asarray( embeddings ) * scale_embedding_weight_value if scale_embedding_weight_value != 1.0 else np.asarray( embeddings )

                # Set CUI-Less Embedding To Values Close To Zero
                for idx in range( self.embeddings_b.shape[1] ): self.embeddings_b[self.concept_id_dictionary[cui_less_token]][idx] = 0.0001

                self.embeddings_b_loaded = True
                generated_embedding_ids  = True

        if generated_embedding_ids: self.generated_embedding_ids = True

        self.Print_Log( "MLPSimilarityDataLoader::Generate_Token_IDs() - Complete" )

    """
        Updates Embedding A and B In Memory Using Token And Concept ID Dictionaries
           Called When Loading Model
    """
    def Update_Token_IDs( self, data_list = [], lowercase = False ):
        # Check(s)
        if self.generated_embedding_ids == True:
            self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Error: Token & Concept Embeddings Have Aleady Been Converted/Formatted" )
            return

        self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        generated_embedding_ids = False

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        # Only Generate Token ID Dictionary Using Embeddings Once.
        #   This Is Skipped During Subsequent Calls To This Function
        if not self.generated_embedding_ids:
            # Generate Embeddings Based On Term Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_a ) > 0:
                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_a ) + 1, len( self.embeddings_a[1].split() ) - 1 ) )

                self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Formatted Embeddings Using Token ID Dictionary" )

                # Parse Embeddings
                for embedding in self.embeddings_a:
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
                        continue

                    # Tokenize Embedding String Into List
                    embedding_elements = embedding.split()

                    # Grab Number Of Embedding Text Elements
                    embedding_idx      = 1

                    # Count Number Of Embedding Elements Found Which Are Non-Numeric
                    for idx, embedding_value in enumerate( embedding_elements ):
                        if not re.search( r'^-*\d+\.\d+', embedding_value ): embedding_idx = idx + 1

                    # If None Are Found, Then Assume The First Element Is Our Embedding Text
                    #   i.e. First Element Is A Numerical Value And The Rest Its Embedding
                    if embedding_idx <= 0: embedding_idx = 1

                    # Join Embedding Text Elements Into Contiguous Text Span Separated By Spaces
                    embedding_text     = " ".join( embedding_elements[0:embedding_idx] )

                    token_id = self.Get_Token_ID( embedding_text )
                    if token_id == -1: token_id = self.Get_Token_ID( embedding_text.lower() )
                    if token_id == -1: continue

                    # Slice Embedding Elements (Float Values)
                    embeddings[token_id] = np.asarray( embedding_elements[embedding_idx:], dtype = 'float32' )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_input_tokens = len( self.token_id_dictionary )

                self.embeddings_a = []
                self.embeddings_a = np.asarray( embeddings )

                self.embeddings_a_loaded = True
                generated_embedding_ids  = True

            # Generate Embeddings Based On Concept Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_b ) > 0:
                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_b ) + 1, len( self.embeddings_b[1].split() ) - 1 ) )

                self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Formatted Embeddings Using Concept ID Dictionary" )

                # Parse Embeddings
                for embedding in self.embeddings_b:
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
                        continue

                    # Tokenize Embedding String Into List
                    embedding_elements = embedding.split()

                    # Grab Number Of Embedding Text Elements
                    embedding_idx      = 1

                    # Count Number Of Embedding Elements Found Which Are Non-Numeric
                    for idx, embedding_value in enumerate( embedding_elements ):
                        if not re.search( r'^-*\d+\.\d+', embedding_value ): embedding_idx = idx + 1

                    # If None Are Found, Then Assume The First Element Is Our Embedding Text
                    #   i.e. First Element Is A Numerical Value And The Rest Its Embedding
                    if embedding_idx <= 0: embedding_idx = 1

                    # Join Embedding Text Elements Into Contiguous Text Span Separated By Spaces
                    embedding_text     = " ".join( embedding_elements[0:embedding_idx] )

                    concept_id = self.Get_Concept_ID( embedding_text )
                    if concept_id == -1: concept_id = self.Get_Concept_ID( embedding_text.lower() )
                    if concept_id == -1: continue

                    # Slice Embedding Elements (Float Values)
                    embeddings[concept_id]  = np.asarray( embedding_elements[embedding_idx:], dtype = 'float32' )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_concept_tokens  = len( self.concept_id_dictionary )

                self.embeddings_b = []
                self.embeddings_b = np.asarray( embeddings )

                # Set CUI-Less Embedding To Values Close To Zero
                if   self.Get_Concept_ID( self.cui_less_token )         != -1: cui_less_token = self.cui_less_token
                elif self.Get_Concept_ID( self.cui_less_token.lower() ) != -1: cui_less_token = self.cui_less_token.lower()
                for idx in range( self.embeddings_b.shape[1] ): self.embeddings_b[self.concept_id_dictionary[cui_less_token]][idx] = 0.0001

                self.embeddings_b_loaded = True
                generated_embedding_ids  = True

        if generated_embedding_ids: self.generated_embedding_ids = True

        self.Print_Log( "MLPSimilarityDataLoader::Update_Token_IDs() - Complete" )

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)
            file_name : File Name (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path, file_name = "" ):
        raise NotImplementedError

        self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Load Necessary Data-Loader Specific Data
        cfg_file_name = "cfg" if file_name == "" else file_name + "_cfg"
        data_loader_settings = self.utils.Read_Data( file_path = file_path + cfg_file_name )

        for setting_and_value in data_loader_settings:
            elements = setting_and_value.split( "<:>" )
            if elements[0] == "Max_Sequence_Length"      : self.Set_Max_Sequence_Length( int( elements[-1] ) )
            if elements[0] == "Number_Of_Inputs"         : self.number_of_input_tokens  = int( elements[-1] )
            if elements[0] == "Number_Of_Outputs"        : self.number_of_output_tokens = int( elements[-1] )

        # Load Token Key ID File
        key_file_name = "key" if file_name == "" else file_name + "_key"
        self.Load_Token_ID_Key_Data( file_path + key_file_name )

        # Load Input/Output Matrices
        input_file_name, output_file_name = "", ""

        if len( file_name ) > 0:
            input_file_name  = str( file_name ) + "_"
            output_file_name = str( file_name ) + "_"

        # Input Matrix
        self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Loading Input Matrix" )

        if self.utils.Check_If_File_Exists( file_path + input_file_name +  "encoded_input.npy" ):
            self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Input Matrix" )
            self.ner_inputs = np.load( file_path + input_file_name +  "encoded_input.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Input Matrix Not Found" )

        # Output Matrix
        self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Loading Output Matrix" )

        if self.utils.Check_If_File_Exists( file_path + output_file_name + "encoded_output.npy" ):
            self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Output Matrix" )
            self.ner_outputs = np.load( file_path + output_file_name + "encoded_output.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Output Matrix Not Found" )

        self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Complete" )

        return False

    """
        Saves Vectorized Model Inputs/Outputs To File.

        Inputs:
            file_path : File Path/Directory (String)
            file_name : File Name (String)

        Outputs:
            None
    """
    def Save_Vectorized_Model_Data( self, file_path, file_name = "" ):
        raise NotImplementedError

        self.Print_Log( "MLPSimilarityDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Save Necessary Data-Loader Specific Data
        data_to_write = "Max_Sequence_Length<:>"        + str( self.Get_Max_Sequence_Length()       ) + "\n"
        data_to_write += "Number_Of_Inputs<:>"          + str( self.Get_Number_Of_Input_Elements()  ) + "\n"
        data_to_write += "Number_Of_Unique_Elements<:>" + str( self.Get_Number_Of_Unique_Tokens()   ) + "\n"
        data_to_write += "Number_Of_Outputs<:>"         + str( self.Get_Number_Of_Unique_Concepts() ) + "\n"
        cfg_file_name = "cfg" if file_name == "" else file_name + "_cfg"
        data_loader_settings = self.utils.Write_Data_To_File( file_path + cfg_file_name, data_to_write )

        # Save Token Key ID File
        key_file_name = "key" if file_name == "" else file_name + "_key"
        self.Save_Token_ID_Key_Data( file_path + key_file_name )

        # Save Input/Output Matrices
        input_file_name, output_file_name = "", ""

        if len( file_name ) > 0:
            input_file_name  = str( file_name ) + "_"
            output_file_name = str( file_name ) + "_"

        # Input Matrix
        self.Print_Log( "MLPSimilarityDataLoader::Save_Vectorized_Model_Data() - Saving Input Matrix" )

        if self.ner_inputs is None:
            self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Warning: Primary Inputs == 'None' / No Data To Save" )
        else:
            np.save( file_path + input_file_name + "encoded_input.npy", self.ner_inputs, allow_pickle = True )

        # Output Matrix
        self.Print_Log( "MLPSimilarityDataLoader::Save_Vectorized_Model_Data() - Saving Output Matrix" )

        if self.ner_outputs is None:
            self.Print_Log( "MLPSimilarityDataLoader::Load_Vectorized_Model_Data() - Warning: Output == 'None' / No Data To Save" )
        else:
            np.save( file_path + output_file_name + "encoded_output.npy", self.ner_outputs, allow_pickle = True )

        self.Print_Log( "MLPSimilarityDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Fetches NER Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        self.Print_Log( "MLPSimilarityDataLoader::Get_Token_ID() - Fetching ID For Token: \"" + str( token ) + "\"" )

        if token in self.token_id_dictionary:
            if self.lowercase_text: token = token.lower()
            self.Print_Log( "MLPSimilarityDataLoader::Get_Token_ID() - Token ID Found: \"" + str( token ) + "\" => " + str( self.token_id_dictionary[token] ) )
            return self.token_id_dictionary[token]
        else:
            self.Print_Log( "MLPSimilarityDataLoader::Get_Token_ID() - Unable To Locate Token In Dictionary" )

        self.Print_Log( "MLPSimilarityDataLoader::Get_Token_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches Concept Token ID From String.

        Inputs:
            concept    : Token (String)

        Outputs:
            concept_id : Token ID Value (Integer)
    """
    def Get_Concept_ID( self, concept ):
        self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_ID() - Fetching ID For Concept: \"" + str( concept ) + "\"" )

        if self.lowercase_text: concept = concept.lower()

        if concept in self.concept_id_dictionary:
            self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_ID() - Token ID Found: \"" + str( concept ) + "\" => " + str( self.concept_id_dictionary[concept] ) )
            return self.concept_id_dictionary[concept]
        else:
            self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_ID() - Unable To Locate Concept In Dictionary" )

        self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches NER Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Token String (String)
    """
    def Get_Token_From_ID( self, index_value ):
        self.Print_Log( "MLPSimilarityDataLoader::Get_Token_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.token_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "MLPSimilarityDataLoader::Get_Token_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "MLPSimilarityDataLoader::Get_Token_From_ID() - Warning: Key Not Found In Dictionary" )

        return None

    """
        Fetches Concept Token String From ID Value.

        Inputs:
            index_value  : Concept ID Value (Integer)

        Outputs:
            key          : Concept TOken String (String)
    """
    def Get_Concept_From_ID( self, index_value ):
        self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.concept_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "MLPSimilarityDataLoader::Get_Concept_From_ID() - Warning: Key Not Found In Dictionary" )

        return None

    """
        Compares The Tokens From The Loaded Data File To The Loaded Embedding Tokens.
            Reports Tokens Present In The Loaded Data Which Are Not Present In The Embedding Representations.

            Note: Load Training/Testing Data, Embeddings And Call self.Generate_Token_IDs() Prior To Calling This Function.

        Inputs:
            None

        Outputs:
            None
    """
    def Generate_Token_Embedding_Discrepancy_Report( self ):
        raise NotImplementedError

        # Check(s)
        if self.Is_Embeddings_A_Loaded() == False:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Error: No Embeddings Loaded In Memory" )
            return

        if self.Is_Data_Loaded() == False:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Error: No Data Loaded In Memory" )
            return

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Locating OOV Tokens / Comparing Data To Embedding Tokens" )

        out_of_vocabulary_tokens = []

        for data in self.Get_Data():
            data_tokens = data.Get_Passage().split()

            for token in data_tokens:
                if self.Get_Token_ID( token ) == -1 and token not in out_of_vocabulary_tokens: out_of_vocabulary_tokens.append( token )

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Generating Discrepancy Report" )

        if len( out_of_vocabulary_tokens ) > 0:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Found " + str( len( out_of_vocabulary_tokens ) ) + " OOV Tokens" )

            report  = "Total Number Of OOV Tokens: " + str( len( out_of_vocabulary_tokens ) ) + "\n"
            report += "OOV Tokens:\n"

            for token in out_of_vocabulary_tokens:
                report += str( token ) + "\t"

            report += "\n"
        else:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - No OOV Tokens Found" )

            report = "No OOV Tokens Found"

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Writing Discrepancy Report To File: \"./discrepancy_report.txt\"" )

        self.utils.Write_Data_To_File( "./discrepancy_report.txt", report )

        # Clean-Up
        report                   = ""
        out_of_vocabulary_tokens = []

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Complete" )


    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################

    def Get_Version( self ):    return self.version


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NERLink.DataLoader import MLPSimilarityDataLoader\n" )
    print( "     data_loader = MLPSimilarityDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

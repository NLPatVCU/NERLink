#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/28/2020                                                                   #
#    Revised: 11/19/2022                                                                   #
#                                                                                          #
#    BioCreative VIII - Task 2 -  Data Loader Class For The NERLink Package.               #
#       Loads BioC Formatted Sequences: Sequences, Labels, Entry Terms, Concept IDs        #
#       And Entry Term Indices For NER and Concept Linking.                                #
#                                                                                          #
#    Also Loads Concept Linking Data From BioCreative V Data-Set. (BC5CDR)                 #
#                                                                                          #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import bioc, re, scipy, threading
import numpy as np
from sparse       import COO, concatenate, load_npz, save_npz
from scipy.sparse import csr_matrix
from bioc         import BioCAnnotation, BioCLocation

# Custom Modules
from NERLink.DataLoader.Base        import DataLoader
from NERLink.DataLoader.BioCreative import Passage


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class BioCreativeDataLoader( DataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  skip_individual_mentions = False, skip_composite_mentions = False, lowercase = False, ignore_label_type_list = [] ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle, lowercase = lowercase,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          skip_individual_mentions = skip_individual_mentions, skip_composite_mentions = skip_composite_mentions,
                          ignore_label_type_list = ignore_label_type_list )
        self.version           = 0.08
        self.annotation_labels = { "O": 0 }

    """
        Reads BIOC Formatted Data
    """
    def Read_Data( self, file_path, lowercase = True, keep_in_memory = True, encode_strings_to_utf8 = True ):
        # Check(s)
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "BioCreativeDataLoader::Read_Data() - Error: File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return self.data_list

        # Store File Path
        self.data_file_path = file_path

        # Store Lowercase Setting In DataLoader Parent Class (Used For self.Get_Token_ID() Function Calls)
        self.lowercase_text = lowercase

        data_list = []

        with open( file_path, 'rb' ) as fp:
            reader = bioc.BioCXMLDocumentReader( fp )

            # Process Each Document
            for document in reader:
                document_id = document.id

                # Iterate Through All Passages In Given Document
                for passage in document.passages:
                    passage_instance        = Passage()

                    annotation_tokens       = []
                    annotation_labels       = []
                    annotation_indices      = []
                    annotation_concept_ids  = []
                    concept_dict            = {}
                    concept_linking_dict    = {}
                    composite_mention_list  = []
                    individual_mention_list = []

                    passage_offset          = passage.offset
                    passage_type            = passage.infons["type"] if "type" in passage.infons else ""
                    passage_text            = passage.text.lower() if lowercase else passage.text
                    passage_text            = self.Clean_Text( passage_text )

                    # Check For Sequences Containing No Data Or Only Containing Whitespace
                    if len( passage_text ) == 0 or len( passage_text.split() ) == 0:
                        self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Passage Contains No Text Data / Skipping Passage" )
                        self.Print_Log( "BioCreativeDataLoader::Read_Data() -     " + str( passage ) )
                        continue

                    if self.max_sequence_length_set == False and len( passage_text.split() ) + 1 > self.max_sequence_length:
                        self.max_sequence_length = len( passage_text.split() ) + 1

                    ###########################################
                    # Add Space Between Last Token And Period #
                    ###########################################
                    last_word = re.findall( r"\b(\w+)\b\.$", passage_text )
                    if last_word:
                        passage_text = re.sub( r"\b(\w+)\b\.$", str( last_word[-1] ) + " .", passage_text  )

                    passage_instance.Set_Passage_Type( passage_type )
                    passage_instance.Set_Passage( passage_text )
                    passage_instance.Set_Passage_Original( passage.text )
                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Document Passage: " + str( passage.text ) )

                    # Parse Through Passage Annotations
                    for annotation in passage.annotations:
                        if annotation.text is not None:
                            annotation_text = annotation.text.lower() if lowercase else annotation.text
                            annotation_text = self.Clean_Text( annotation_text )

                            self.Print_Log( "BioCreativeDataLoader::Read_Data() - Annotation Text: " + str( annotation_text ) )

                        # Parse Annotation Type
                        if "type" in annotation.infons:
                            annotation_type = annotation.infons["type"].lower()

                            # Skip Annotation Types Specified in Ignore Annotation Label Type List
                            if annotation_type in self.ignore_label_type_list:
                                self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Omitting Annotation / Annotation Type Exists In 'self.ignore_label_type_list' List" )
                                continue
                            # Parse Annotation Location If Annotation Type == 'Chemical', 'Disease', etc.
                            elif annotation_type in self.cl_accepted_labels:
                                # Check
                                if annotation_text is None:
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Annotation Contains No Text / Skipping Annotation" )
                                    continue

                                self.Print_Log( "BioCreativeDataLoader::Read_Data() - Type: " + str( annotation.infons["type"] ) )
                                annotation_label = annotation.infons["type"].lower() if lowercase else annotation.infons["type"]

                                # Add Annotation Label To Annotation Type Labels List
                                if annotation_label not in annotation_type_labels:
                                    annotation_type_labels.append( annotation_label )

                                # Add Annotation Data To Passage Instance Lists
                                annotation_tokens.append( annotation_text )
                                annotation_labels.append( annotation_label )

                                # Note: There are two types of 'CompositeRole': 'CompositeMention' and 'IndividualMention'.
                                #       'CompositeMention'  : Contains All Entity Mentions Into A Single Entity Mention
                                #       'IndividualMention' : Contains An Individual Entity Mention Within A 'CompositeMention'
                                if "CompositeRole" in annotation.infons:
                                    if annotation.infons["CompositeRole"].lower() == "compositemention":
                                        composite_mention_list.append( True )
                                        individual_mention_list.append( False )
                                    elif annotation.infons["CompositeRole"].lower() == "individualmention":
                                        composite_mention_list.append( False )
                                        individual_mention_list.append( True )
                                else:
                                    composite_mention_list.append( False )
                                    individual_mention_list.append( False )

                                annotation_index_instances = ""

                                for count, location in enumerate( annotation.locations ):
                                    start_index = location.offset - passage_offset
                                    end_index   = location.end - passage_offset
                                    annotation_index_instances += str( start_index ) + ":" + str( end_index )
                                    if count < len( annotation.locations ) - 1: annotation_index_instances += "<:>"
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Location Indices: " + str( start_index ) + " to " + str( end_index ) )

                                annotation_indices.append( annotation_index_instances )

                                # Add Annotation Concept ID
                                if "identifier" not in annotation.infons and "MESH" not in annotation.infons and "CUI" not in annotation.infons:
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Annotation Contains No Linking Concept ID Or Concept Linking Type Unknown", force_print = True )
                                    self.Print_Log( "                                   - Annotation Data: " + str( annotation.infons ) )
                                    continue

                                annotation_concept_id, is_mesh, is_cui = None, False, False

                                # BioCreative VII (BC7T2) - Chemical Mentions
                                if "identifier" in annotation.infons:
                                    is_mesh = True
                                    annotation_concept_id = annotation.infons["identifier"].lower() if lowercase else annotation.infons["identifier"]
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Concept Type: \"idenifier\" - Concept ID: " + str( annotation_concept_id ) )
                                # BioCreative V (BC5CDR) - Chemical & Disease Mentions
                                elif "MESH" in annotation.infons:
                                    is_mesh = True
                                    annotation_concept_id = annotation.infons["MESH"].lower()       if lowercase else annotation.infons["MESH"]
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Concept Type: \"MESH\" - Concept ID: " + str( annotation_concept_id ) )
                                elif "CUI" in annotation.infons:
                                    is_cui = True
                                    annotation_concept_id = annotation.infons["CUI"].lower()        if lowercase else annotation.infons["CUI"]
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Concept Type: \"CUI\" - Concept ID: " + str( annotation_concept_id ) )

                                # Try And Capture "-" or "-1" / CUI-LESS Annotation And Set Accordingly
                                if annotation_concept_id in ["-", "-1"]:
                                    annotation_concept_id = self.Get_CUI_LESS_Token().lower() if lowercase else self.Get_CUI_LESS_Token()

                                # Determine If Multiple MESH IDs Are Within The 'annotation_concept_id' But Separated By '|' Character
                                #   If So, Separate Them By The Task Default Character Containing Multiple MESH ID
                                #     i.e 'mesh:d014527|d012492' -> 'mesh:d014527,mesh:d012492'
                                if "|" in annotation_concept_id:
                                    concept_id_prefix = None

                                    if is_mesh:
                                        if not re.match( r'^[Mm][Ee][Ss][Hh]', annotation_concept_id ):
                                            concept_id_prefix = ","
                                        else:
                                            concept_id_prefix = ",mesh:" if lowercase else ",MESH:"
                                    else:
                                        concept_id_prefix = ","

                                    annotation_concept_id = re.sub( r'\|', concept_id_prefix, annotation_concept_id )

                                # Remove Preceding And Trailing Whitespace
                                annotation_concept_id = re.sub( r'^\s+|\s+$', "", annotation_concept_id )

                                annotation_concept_ids.append( annotation_concept_id )
                                if annotation_concept_id not in concept_linking_dict: concept_linking_dict[annotation_concept_id] = annotation_text

                            # Parse Concept Normalization Terms & Linked MeSH Concepts If Annotation Type == "MeSH_Indexing_Chemical"
                            #    This Assumes Each 'MeSH Concept' Has An Associated 'Entry Term'.
                            #    No Checking Is Performed To Ensure This Is The Case.
                            elif annotation.infons["type"].lower() == "mesh_indexing_chemical":
                                # Check(s)
                                if "identifier" not in annotation.infons:
                                    self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Entry Term Contains No Concept Linking MeSH ID" )
                                    continue

                                entry_term = annotation.infons["entry_term"].lower() if lowercase else annotation.infons["entry_term"]
                                identifer  = annotation.infons["identifier"].lower() if lowercase else annotation.infons["identifier"]

                                # Remove Preceding And Trailing Whitespace
                                entry_term = re.sub( r'^\s+|\s+$', "", entry_term )
                                identifer  = re.sub( r'^\s+|\s+$', "", identifer )

                                concept_dict[entry_term] = identifer

                                # Keep Track Of Unique MeSH Identifers. Value = Concept Frequency (Not Used)
                                if identifer not in self.concept_frequency_dictionary:
                                    self.concept_frequency_dictionary[identifer] = 1
                                else:
                                    self.concept_frequency_dictionary[identifer] += 1

                                self.Print_Log( "BioCreativeDataLoader::Read_Data() -       Indexing Term: " + str( entry_term ) + " => Concept ID: " + str( identifer ) )
                            # Warn User If New Annotation Type Has Been Found (Unknown Annotation Type)
                            else:
                                self.Print_Log( "BioCreativeDataLoader::Read_Data() -  Warning: New Annotation Type - " + str( annotation.infons["type"] ), force_print = True )
                        elif annotation.infons and annotation.infons.lower() not in ["type", "entry_term", "identifier", "compositemention", "individualmention"]:
                            self.Print_Log( "BioCreativeDataLoader::Read_Data() -  Warning: New Annotation Key - " + str( annotation.infons ), force_print = True )
                        else:
                            self.Print_Log( "BioCreativeDataLoader::Read_Data() - Warning: Annotation Contains No Type" )

                    # ToDo - Parse Through Passage Relations / Currently Not Used
                    for relation in passage.relations:
                        pass

                    # ToDo - Parse Through Sentences / Currently Not Used
                    for sentence in passage.sentences:
                        pass

                    passage_instance.Set_Document_ID( document_id )
                    passage_instance.Set_Annotations( annotation_tokens )
                    passage_instance.Set_Annotation_Labels( annotation_labels )
                    passage_instance.Set_Annotation_Indices( annotation_indices )
                    passage_instance.Set_Annotation_Concept_IDs( annotation_concept_ids )
                    passage_instance.Set_Concepts( concept_dict )
                    passage_instance.Set_Concept_Linking( concept_linking_dict )
                    passage_instance.Set_Composite_Mention_List( composite_mention_list )
                    passage_instance.Set_Individual_Mention_List( individual_mention_list )

                    # Add Document To Collection
                    data_list.append( passage_instance )

            # Set Max Sequence Length Variable Set Flag == True.
            #   This Way No Other Data File Reads Will Reset The Variable Value
            self.max_sequence_length_set = True

        if keep_in_memory:
            self.Print_Log( "BioCreativeDataLoader::Read_Data() - Storing Processed Data In Memory" )
            self.data_list = data_list

        # Generate Complete Annotation Type List (NER Labels)
        if len( annotation_type_labels ) > 0:
            annotation_type_labels = sorted( annotation_type_labels )
            annotation_idx         = len( self.annotation_labels ) - 1

            # Add New Annotation Type Labels
            for annotation_type_label in annotation_type_labels:
                b_annotation_type_label = "B-" + str( annotation_type_label )
                i_annotation_type_label = "I-" + str( annotation_type_label )

                if b_annotation_type_label not in self.annotation_labels.keys():
                    self.annotation_labels[b_annotation_type_label] = annotation_idx
                    annotation_idx += 1
                if i_annotation_type_label not in self.annotation_labels.keys():
                    self.annotation_labels[i_annotation_type_label] = annotation_idx
                    annotation_idx += 1

            # Adjust "<*>PADDING<*>" Label
            self.annotation_labels[self.Get_Padding_Token()] = annotation_idx

        self.Print_Log( "BioCreativeDataLoader::Read_Data() - Complete" )

        return data_list

    """
        Vectorized/Binarized NER Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list          : List Of Passage Objects
            use_csr_format     : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            keep_in_memory     : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            number_of_threads  : Number Of Threads To Deploy For Data Vectorization (Integer)
            is_validation_data : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)

        Outputs:
            ner_inputs         : CSR Matrix or Numpy Array
            ner_outputs        : CSR, COO Matrix or Numpy Array
    """
    def Encode_NER_Model_Data( self, data_list = [], use_csr_format = False, keep_in_memory = True, number_of_threads = 4, is_validation_data = False, is_evaluation_data = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        if number_of_threads < 1:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Number Of Threads < 1 / Setting Number Of Threads = 1", force_print = True )
            number_of_threads = 1

        threads     = []
        ner_inputs  = []
        ner_outputs = []

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Vectorizing Data Using Settings" )
        self.Print_Log( "                                           - Use CSR Format    : " + str( use_csr_format ) )

        total_number_of_instances = len( data_list )

        if number_of_threads > total_number_of_instances:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: 'number_of_threads > len( data_list )' / Setting 'number_of_threads = total_number_of_instances'" )
            number_of_threads = total_number_of_instances

        instances_per_thread = int( ( total_number_of_instances + number_of_threads - 1 ) / number_of_threads )

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Number Of Threads: " + str( number_of_threads ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Instances Per Thread : " + str( instances_per_thread  ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Total Instances In File Data: " + str( total_number_of_instances ) )

        ###########################################
        #          Start Worker Threads           #
        ###########################################

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Starting Worker Threads" )

        # Create Storage Locations For Threaded Data Segments
        tmp_thread_data = [None for i in range( number_of_threads )]

        for thread_id in range( number_of_threads ):
            starting_instance_index = instances_per_thread * thread_id
            ending_instance_index   = starting_instance_index + instances_per_thread if starting_instance_index + instances_per_thread < total_number_of_instances else total_number_of_instances

            new_thread = threading.Thread( target = self.Worker_Thread_Function, args = ( thread_id, data_list[starting_instance_index:ending_instance_index], tmp_thread_data, use_csr_format ) )
            new_thread.start()
            threads.append( new_thread )

        ###########################################
        #           Join Worker Threads           #
        ###########################################

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Waiting For Worker Threads To Finish" )

        for thread in threads:
            thread.join()

        # Convert To CSR Matrix Format
        if use_csr_format:
            ner_inputs  = csr_matrix( ner_inputs  )
            ner_outputs = COO( ner_outputs )

        if len( tmp_thread_data ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error Vectorizing Model Data / No Data Returned From Worker Threads", force_print = True )
            return None, None

        # Concatenate Vectorized Model Data Segments From Threads
        for model_data in tmp_thread_data:
            if model_data is None or len( model_data ) < 2:
                self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error: Expected At Least Two Vectorized Lists From Worker Threads / Received None Or < 2", force_print = True )
                continue

            # Vectorized Inputs/Outputs
            ner_encoded_input  = model_data[0]
            ner_encoded_output = model_data[1]

            ###############################################
            # Convert Primary Inputs To CSR Matrix Format #
            ###############################################
            if use_csr_format:
                # CSR Matrices Are Empty, Overwrite Them With The Appropriate Data With The Correct Shape
                if ner_inputs.shape[1] == 0:
                    ner_inputs = ner_encoded_input
                # Stack Existing CSR Matrices With New Data By Row
                else:
                    # In-Place Update (Should Be Faster Than The New Copy Replacement)
                    ner_inputs.data    = np.hstack( ( ner_inputs.data, ner_encoded_input.data ) )
                    ner_inputs.indices = np.hstack( ( ner_inputs.indices, ner_encoded_input.indices ) )
                    ner_inputs.indptr  = np.hstack( ( ner_inputs.indptr, ( ner_encoded_input.indptr + ner_inputs.nnz )[1:] ) )
                    ner_inputs._shape  = ( ner_inputs.shape[0] + ner_encoded_input.shape[0], ner_encoded_input.shape[1] )
            else:
                for i in range( len( ner_encoded_input ) ):
                    ner_inputs.append( ner_encoded_input[i] )

            ########################################
            # Convert Outputs To CSR Matrix Format #
            ########################################
            if use_csr_format:
                # COO Matrix Is Empty, Overwrite Them With The New COO Matrix
                if len( ner_outputs.shape ) == 0:
                    ner_outputs = ner_encoded_output
                # Concatenate Existing COO Matrices With New Data
                else:
                    ner_outputs = concatenate( [ner_outputs, ner_encoded_output], axis = 0 )
            else:
                for i in range( len( ner_encoded_output ) ):
                    ner_outputs.append( ner_encoded_output[i] )

        if use_csr_format == False:
            ner_inputs  = np.asarray( ner_inputs  )
            ner_outputs = np.asarray( ner_outputs )

        if isinstance( ner_inputs, list ):
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Input Length  : " + str( len( ner_inputs ) ) )
        elif isinstance( ner_inputs, csr_matrix ) or isinstance( ner_inputs, np.ndarray ):
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Input Length  : " + str( ner_inputs.shape  ) )

        if isinstance( ner_outputs, list ):
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Output Length : " + str( len( ner_outputs ) ) )
        elif isinstance( ner_outputs, COO ) or isinstance( ner_inputs, np.ndarray ):
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Output Length : " + str( ner_outputs.shape  ) )

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Vectorized Inputs  :\n" + str( ner_inputs  ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Vectorized Outputs :\n" + str( ner_outputs ) )

        # Clean-Up
        threads         = []
        tmp_thread_data = []

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Complete" )

        #####################
        # List Final Checks #
        #####################
        if isinstance( ner_inputs, list ) and len( ner_inputs ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_outputs, list ) and len( ner_outputs ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        ######################
        # Array Final Checks #
        ######################
        if isinstance( ner_inputs, np.ndarray ) and ner_inputs.shape[0] == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_outputs, np.ndarray ) and ner_outputs.shape[0] == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        ###########################
        # CSR Matrix Final Checks #
        ###########################
        if isinstance( ner_inputs, csr_matrix ) and ner_inputs.nnz == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_outputs, COO ) and ner_outputs.nnz == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        # These Can Be Saved Via DataLoader::Save_Vectorized_Model_Data() Function Call.
        if keep_in_memory:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Model_Data() - Storing In Memory" )

            if is_validation_data:
                self.ner_val_inputs   = ner_inputs
                self.ner_val_outputs  = ner_outputs
            elif is_evaluation_data:
                self.ner_eval_inputs  = ner_inputs
                self.ner_eval_outputs = ner_outputs
            else:
                self.ner_inputs       = ner_inputs
                self.ner_outputs      = ner_outputs

        return ner_inputs, ner_outputs

    """
        Encodes Concept Linking Data - Used For Training, Validation Or Evaluation Data

        Inputs:
            data_list          : List Of Passage Objects
            use_csr_format     : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            keep_in_memory     : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            is_validation_data : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
            pad_output         : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                 Categorical Crossentropy vs. Sparse Categorical Crossentropy
            concept_delimiter  : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                 Used For One-To-Many Relationships

        Outputs:
            concept_inputs     : Numpy Array
            concept_outputs    : CSR, COO Matrix or Numpy Array
    """
    def Encode_CL_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = True, keep_in_memory = True,
                              is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                              mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = False,
                              use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        # Clear Previous Concept Instance Data Index List
        self.concept_instance_data_idx.clear()

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Encoding Concept Instances" )

        encoded_concept_inputs, encoded_concept_outputs = [], []
        output_row, output_col, output_data             = [], [], []
        output_row_index, passage_id                    = 0, 0

        for passage in data_list:
            # Encode All Term and Concept Pairs
            for annotation_tokens, annotation_concepts, is_individual_mention, is_composite_mention in zip( passage.Get_Annotations(), passage.Get_Annotation_Concept_IDs(),
                                                                                                            passage.Get_Composite_Mention_List(), passage.Get_Individual_Mention_List() ):
                if annotation_tokens == "" or annotation_concepts == "":
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Warning: Instance Contains No Entry Terms Or Concept IDs" )
                    self.Print_Log( "                                                       - Sequence: " + str( passage.Get_Passage() ) )
                    continue
                elif is_composite_mention and self.skip_composite_mentions or "," in annotation_tokens:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Composite Mention Detected / Skipping Composite Mention" )
                    continue
                elif is_individual_mention and self.skip_individual_mentions:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Individual Mention Detected / Skipping Individual Mention" )
                    continue

                encoded_input_instance, encoded_output_instance = self.Encode_CL_Instance( entry_term = annotation_tokens, annotation_concept = annotation_concepts,
                                                                                           pad_output = pad_output, concept_delimiter = concept_delimiter )

                if encoded_input_instance is None or encoded_output_instance is None:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Input And/Or Output Instance" )

                    # Keep Track Of OOV Terms And Frequency
                    if annotation_tokens not in self.oov_term_dict:
                        self.oov_term_dict[annotation_tokens] = 1
                    else:
                        self.oov_term_dict[annotation_tokens] += 1
                    continue

                # Concept Input Is A Single Index i.e. [14]
                encoded_concept_inputs.append( encoded_input_instance )

                # Concept Output Is A Vector/Array Of 'N' Classes With Our Desired Instance Class As '1'
                #   i.e. [0, 0, 1, 0, 0]
                if use_csr_format:
                    for i, value in enumerate( encoded_output_instance ):
                        if value == 0: continue

                        output_row.append( output_row_index )
                        output_col.append( i )
                        output_data.append( value )

                    output_row_index += 1
                else:
                    encoded_concept_outputs.append( encoded_output_instance )

                # Keep Track Of Which Passage The Instance Came From
                self.concept_instance_data_idx.append( passage_id )

            passage_id += 1

        # Convert Data To Numpy Arrays
        encoded_concept_inputs = np.asarray( encoded_concept_inputs, dtype = np.int32 )

        # Convert Into CSR_Matrix
        if use_csr_format:
            output_data             = np.asarray( output_data, dtype = np.int32 )
            number_of_output_rows   = self.Get_Number_Of_Unique_Concepts() if pad_output else 1
            encoded_concept_outputs = COO( [ output_row, output_col ], output_data, shape = ( output_row_index, number_of_output_rows ), fill_value = 0 )
        else:
            encoded_concept_outputs = np.asarray( encoded_concept_outputs, dtype = np.int32 )

        # Check(s)
        number_of_input_instances  = len( encoded_concept_inputs )
        number_of_output_instances = encoded_concept_outputs.shape[0] if isinstance( encoded_concept_outputs, COO ) else len( encoded_concept_outputs )

        if number_of_input_instances == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Inputs", force_print = True )
            return None, None
        elif number_of_output_instances == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Outputs", force_print = True )
            return None, None
        elif number_of_input_instances != number_of_output_instances:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Error: Number Of Input And Output Instances Not Equal", force_print = True )
            return None, None

        if keep_in_memory:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Storing Encoded Data In Memory" )

            if is_validation_data:
                self.concept_val_inputs   = encoded_concept_inputs
                self.concept_val_outputs  = encoded_concept_outputs
            elif is_evaluation_data:
                self.concept_eval_inputs  = encoded_concept_inputs
                self.concept_eval_outputs = encoded_concept_outputs
            else:
                self.concept_inputs       = encoded_concept_inputs
                self.concept_outputs      = encoded_concept_outputs

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Complete" )

        return encoded_concept_inputs, encoded_concept_outputs

    """
        Tokenizes Data Sequences Into List Of Tokens With Or Without Padding
            Used For ELMo Implementation

        Input:
            data_list              : List Of Variables
            use_padding            : Pads Each Sequence To 'self.max_sequence_length' Token Elements

        Output:
    """
    def Tokenize_Model_Data( self, data_list = [], use_padding = True ):
        # If The User Does Not Pass Any Data, Try The Data Stored In Memory (DataLoader Object)
        if len( data_list ) == 0 and len( self.data_list ) > 0:
            self.Print_Log( "BioCreativeDataLoader::Tokenize_Model_Data() - Warning: No Data Specified, Using Data Stored In Memory" )
            data_list = self.data_list

        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Tokenize_Model_Data() - Error: Data List Is Empty", force_print = True )
            return []

        self.Print_Log( "BioCreativeDataLoader::Tokenize_Model_Data() - Tokenizing Data Sequences" )
        self.Print_Log( "BioCreativeDataLoader::Tokenize_Model_Data() -   Use Padding: " + str( use_padding ) )

        tokenized_sequences = []

        # Split Each Passage Text Sequence By White Space And Store As List Of Tokens
        for passage in data_list:
            passage_length = len( passage.Get_Passage().split() )
            sequence = [ "" for _ in range( self.Get_Max_Sequence_Length() ) ] if use_padding else [ "" for _ in range( passage_length ) ]

            for index, token in enumerate( passage.Get_Passage().split() ):
                if index >= self.Get_Max_Sequence_Length(): break
                sequence[index] = token

            # Store Tokenized Sequence
            tokenized_sequences.append( sequence )

        self.Print_Log( "BioCreativeDataLoader::Tokenize_Model_Data() - Complete" )

        return tokenized_sequences

    """
        Returns List Of Strings, Compiled From Data Sequences
            Used For ELMo Implementation

        Input:
            data_list              : List Of Variables

        Output:
    """
    def Get_Data_Sequences( self, data_list = [] ):
        # If The User Does Not Pass Any Data, Try The Data Stored In Memory (DataLoader Object)
        if len( data_list ) == 0 and len( self.data_list ) > 0:
            self.Print_Log( "BioCreativeDataLoader::Get_Data_Sequences() - Warning: No Data Specified, Using Data Stored In Memory" )
            data_list = self.data_list

        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Get_Data_Sequences() - Error: Data List Is Empty", force_print = True )
            return []

        self.Print_Log( "BioCreativeDataLoader::Get_Data_Sequences() - Fetching Data Sequences" )

        # Split Each Passage Text Sequence By White Space And Store As List Of Tokens
        sequences = [passage.Get_Passage() for passage in data_list]

        self.Print_Log( "BioCreativeDataLoader::Get_Data_Sequences() - Complete" )

        return sequences

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            text_sequence               : Sequence Of Text

        Outputs:
            encoded_text_sequence       : Encoded Text Sequence (List/Vector Of Integers)
            encoded_annotation_sequence : Encoded Sequence Token Labels (List/Vector Of Integers)
    """
    def Encode_NER_Instance( self, text_sequence, annotations, annotation_labels, annotation_indices, composite_mention_list = [], individual_mention_list = [] ):
        # Check
        if len( self.token_id_dictionary ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Error: Token ID Dictionary Is Empty", force_print = True )
            return [], []

        # Placeholders For Vectorized Inputs/Outputs (Pad Remaining Tokens Outside Of Sequence Length)
        encoded_text_sequence       = [ 0 for _ in range( self.Get_Max_Sequence_Length() ) ]
        encoded_annotation_sequence = [ np.asarray( [ 0 if idx != 0 else 1 for idx in range( len( self.annotation_labels ) ) ] )
                                        for _ in range( self.Get_Max_Sequence_Length() ) ]

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Encoding Inputs" )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -                   Text Sequence         : " + str( text_sequence         ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -                   Annotations           : " + str( annotations           ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -                   Annotation Indices    : " + str( annotation_indices    ) )

        # Encode Text Sequence
        for index, token in enumerate( text_sequence.split() ):
            if index >= self.max_sequence_length - 1: continue

            token_id = self.Get_Token_ID( token )

            if token_id == -1:
                self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Error: Token \"" + str( token ) + "\" Does Not Exist Within Dictionary / OOV Term" )
                return [], []

            encoded_text_sequence[index] = token_id

        text_sequence_character_mask = [ "_" if text_sequence[i] != " " else " " for i in range( len( text_sequence ) ) ]

        # Determine Which Entities Have Annotations
        curr_annotation_index = 0

        for annotation, annotation_label, annotation_idx, is_composite_mention, is_individual_mention in zip( annotations, annotation_labels, annotation_indices,
                                                                                                              composite_mention_list, individual_mention_list ):
            # Skip Individual Mentions - There Are Contained Within Composite Mentions
            if is_individual_mention: continue

            number_of_indices = len( annotation_idx.split( "<:>" ) )

            for indices in annotation_idx.split( "<:>" ):
                indices           = indices.split( ":" )
                annotation_offset = int( indices[0]  )
                annotation_end    = int( indices[-1] )

                # Extract Annotation Token Using Token Indices (Debugging Testing)
                extracted_token   = text_sequence[annotation_offset:annotation_end]

                if number_of_indices == 1 and annotation != extracted_token:
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Error: Extracted Token != True Token", force_print = True )
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -        True Token: " + str( annotation ), force_print = True )
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -        Extracted Token: " + str( extracted_token ), force_print = True )
                    continue
                elif number_of_indices > 1 and extracted_token not in annotation:
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Error: Extracted Token Not In True Token", force_print = True )
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -        True Token: " + str( annotation ), force_print = True )
                    self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() -        Extracted Token: " + str( extracted_token ), force_print = True )
                    continue

                index_size = len( str( curr_annotation_index ) )
                text_sequence_character_mask[annotation_offset:annotation_offset+index_size] = str( curr_annotation_index )
                text_sequence_character_mask[annotation_offset + index_size:annotation_end]    = ["#"] * ( annotation_end - annotation_offset - index_size )

                # Double Check To See If The Annotation Doesn't Contain Spaces (Annotation Not Composed Of Multiple Singular Terms)
                space_indices = [i for i, char in enumerate( annotation ) if char == " "]

                # Fill Spaces Back In
                for index in space_indices:
                    offset_index = annotation_offset + index
                    text_sequence_character_mask[offset_index]     = " "
                    text_sequence_character_mask[offset_index + 1] = str( curr_annotation_index )

            curr_annotation_index += 1

        text_sequence_character_mask     = '' . join( text_sequence_character_mask )
        text_sequence_annotation_indices = text_sequence_character_mask.split()

        # Encoding Error / No Annotations Matched In Mask
        if len( [ True for token in text_sequence_annotation_indices if re.search( r'\d+', token ) is not None ] ) == 0: return [], []

        # Set Annotation Indices In 'encoded_annotation_sequence' Variable
        processed_indices  = []

        for index, text_sequence_annotation_index in enumerate( text_sequence_annotation_indices ):
            if index >= self.max_sequence_length - 1: continue
            curr_annotation_index = re.sub( r'\_|\#', "", text_sequence_annotation_index )

            # Set Current Entity As Beginning Or Intermediate Of Entity Label
            if len( curr_annotation_index ) > 0:
                annotation_label_index = -1

                if curr_annotation_index not in processed_indices:
                    annotation_label_index = int( self.annotation_labels["B-Chemical"] )
                else:
                    annotation_label_index = int( self.annotation_labels["I-Chemical"] )

                # Fill The Non-Chemical Tokens With "O" Token
                if annotation_label_index != -1:
                    encoded_annotation_sequence[index][self.annotation_labels["O"]] = 0
                    encoded_annotation_sequence[index][annotation_label_index]      = 1

                processed_indices.append( curr_annotation_index )
            # Entity Has No Label, Designate "O" Token
            else:
                encoded_annotation_sequence[index][self.annotation_labels["O"]] = 1

        self.Print_Log( "BioCreativeDataLoader::Encode_NER_Instance() - Complete" )

        return encoded_text_sequence, encoded_annotation_sequence

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            entry_term         : Concept Token (String)

        Outputs:
            encoded_entry_term : Encoded Entry Term Vector
    """
    def Encode_CL_Input_Instance( self, entry_term, text_sequence = None, annotation_indices = None, pad_input = True,
                                  mask_term_sequence = False, separate_sentences = True, term_sequence_only = False,
                                  restrict_context = False ):
        # Check
        if self.Get_Number_Of_Unique_Tokens() == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Error: Token ID Dictionary Is Empty", force_print = True )
            return None

        encoded_entry_term = []

        # Map Entry Terms To Concepts
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Input_Instance() - Encoding Entry Term" )
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Input_Instance() -      Entry Term: " + str( entry_term ) )

        # Encode Entry Term
        token_id = self.Get_Token_ID( entry_term )

        if token_id == -1:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Input_Instance() - Error: Token \"" + str( entry_term ) + "\" Does Not Exist Within Dictionary / OOV Term" )
            return None

        encoded_entry_term.append( token_id )

        # Check
        if -1 in encoded_entry_term:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Input_Instance() - Warning: Entry Term \'" + str( entry_term ) + "\' Not In Dictionary", force_print = True )
            return None

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Input_Instance() -      Encoded Entry Term: " + str( encoded_entry_term ) )

        return encoded_entry_term

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            annotation_concept : Concept Token MeSH ID / CUI (String)
            concept_delimiter  : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                 Used For One-To-Many Relationships
            pad_output         : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                 Categorical Crossentropy vs. Sparse Categorical Crossentropy

        Outputs:
            encoded_concept    : Encoded Concept Vector (List/Vector Of Integers)
    """
    def Encode_CL_Output_Instance( self, annotation_concept, concept_delimiter = None, pad_output = False, label_per_sub_word = False ):
        # Check
        if len( self.concept_id_dictionary ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None

        if label_per_sub_word:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: 'label_per_sub_word = True' / Not Supported Setting To 'False'", force_print = True )
            label_per_sub_word = False

        encoded_concept = [ 0 for _ in range( self.Get_Number_Of_Unique_Concepts() ) ] if pad_output else []

        # Map Entry Terms To Concepts
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Encoding Entry Concept To Concept ID" )
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() -      Concept   : " + str( annotation_concept ) )

        # Encode Concept(s)
        concept_id = None

        if concept_delimiter is not None:
            for concept in annotation_concept.split( concept_delimiter ):
                concept_id = self.Get_Concept_ID( concept )

                if concept_id == -1:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( annotation_concept ) + "\' Not In Dictionary", force_print = True )

                if pad_output:
                    encoded_concept[concept_id] = 1
                else:
                    encoded_concept.append( concept_id )
        else:
            concept_id = self.Get_Concept_ID( annotation_concept )

            if concept_id == -1:
                self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( annotation_concept ) + "\' Not In Dictionary", force_print = True )
                return None

            if pad_output:
                encoded_concept[concept_id] = 1
            else:
                encoded_concept.append( concept_id )

        # Check To See If Any Concept IDs Were Encoded
        if pad_output and 1 not in encoded_concept:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Error: No Concept IDs Were Encoded For Multi-Output Instance", force_print = True )
            return None
        elif pad_output == False and len( encoded_concept ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() - Error: No Concept IDs Were Encoded For Multi-Output Instance", force_print = True )
            return None

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Output_Instance() -      Encoded Concept   : " + str( encoded_concept ) )

        return encoded_concept

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
        # Check(s)
        if self.Get_Number_Of_Unique_Tokens() == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Error: Token ID Dictionary Is Empty", force_print = True )
            return None, None

        if self.Get_Number_Of_Unique_Concepts() == 0:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None, None

        # Map Entry Terms To Concepts
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Encoding Entry Term/Concept To Concept ID" )

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() -      Entry Term: " + str( entry_term   ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() -      Concept   : " + str( annotation_concept ) )

        encoded_entry_term = self.Encode_CL_Input_Instance( entry_term = entry_term, text_sequence = text_sequence, annotation_indices = annotation_indices,
                                                            pad_input = pad_input, mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                            term_sequence_only = term_sequence_only, restrict_context = restrict_context )

        if encoded_entry_term is None:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Error Encoding Term: " + str( entry_term ) )
            return None, None

        encoded_concept = self.Encode_CL_Output_Instance( annotation_concept = annotation_concept, concept_delimiter = concept_delimiter,
                                                          pad_output = pad_output, label_per_sub_word = label_per_sub_word )

        if encoded_concept is None:
            self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() - Error Encoding Concept: " + str( annotation_concept ) )
            return None, None

        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() -      Encoded Entry Term: " + str( encoded_entry_term ) )
        self.Print_Log( "BioCreativeDataLoader::Encode_CL_Instance() -      Encoded Concept   : " + str( encoded_concept    ) )

        return encoded_entry_term, encoded_concept

    """
        Decodes Sequence Of Token IDs to Sequence Of Token Strings

        Inputs:
            text_sequence               : Sequence Of Token IDs
            remove_padding              : Removes Padding Tokens From Returned String Sequence

        Outputs:
            decoded_text_sequence       : Decoded Text Sequence (List Of Strings)
    """
    def Decode_NER_Input_Instance( self, encoded_text_sequence, remove_padding = False ):
        # Check(s)
        if isinstance( encoded_text_sequence, list ) and len( encoded_text_sequence ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Input_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_text_sequence, np.ndarray ) and encoded_text_sequence.shape[0] == 0 or \
           isinstance( encoded_text_sequence, COO        ) and encoded_text_sequence.shape[0] == 0 or \
           isinstance( encoded_text_sequence, csr_matrix ) and encoded_text_sequence.shape[0] == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Input_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        # Note: Only The First Encoded Sequence Will Be Decoded
        if isinstance( encoded_text_sequence, np.ndarray ) and encoded_text_sequence.ndim == 2:
            encoded_text_sequence = np.asarray( encoded_text_sequence )[0]

        if isinstance( encoded_text_sequence, COO ) and encoded_text_sequence.ndim == 2:
            encoded_text_sequence = encoded_text_sequence.todense()[0]

        if isinstance( encoded_text_sequence, csr_matrix ) and encoded_text_sequence.ndim == 2:
            encoded_text_sequence = encoded_text_sequence.todense()[0]

        self.Print_Log( "BioCreativeDataLoader::Decode_NER_Input_Instance() - Encoded Token ID Sequence: " + str( encoded_text_sequence ) )

        decoded_text_sequence = []

        if remove_padding:
            decoded_text_sequence = [ self.Get_Token_From_ID( token_id ) for token_id in encoded_text_sequence if token_id != self.Get_Token_ID( self.padding_token ) ]
        else:
            decoded_text_sequence = [ self.Get_Token_From_ID( token_id ) for token_id in encoded_text_sequence ]

        self.Print_Log( "BioCreativeDataLoader::Decode_NER_Input_Instance() - Decoded Token Sequence: " + str( decoded_text_sequence ) )

        return decoded_text_sequence

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings
    """
    def Decode_NER_Output_Instance( self, encoded_output_sequence ):
        # Check(s)
        if isinstance( encoded_output_sequence, list ) and len( encoded_output_sequence ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, COO        ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.shape[0] == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = np.asarray( encoded_output_sequence )[0]

        if isinstance( encoded_output_sequence, COO ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        if isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Encoded Output ID Sequence: " + str( encoded_output_sequence ) )

        idx_to_tokens = { v:k for k,v in self.Get_Annotation_Labels().items() }

        # Find The Index Of The Largest Element For Each Prediction Array/List And Use It's Index To Obtain The Corresponding Annotation Token
        decoded_output_sequence = [ idx_to_tokens[np.argmax( prediction_array )] for prediction_array in encoded_output_sequence ]

        self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Decoded Output Sequence: " + str( decoded_output_sequence ) )

        return decoded_output_sequence

    """
        Decodes Input & Output Sequence Of NER Token IDs And Label IDs To Sequence Of NER Token & Label Strings
    """
    def Decode_NER_Instance( self, encoded_output_sequence, encoded_input_sequence = None, remove_padding = True ):
        self.Print_Log( "BioCreativeDataLoader::Decode_NER_Instance() - Not Implemented" )
        raise NotImplementedError
        return [], []

    """
        Decodes Input Sequence Instance Of IDs To Entry Term String(s)

        Note: This Assume Only One Entry Term Exists Per Input Sequence.

        Inputs:
            encoded_input_instance : List/Numpy Array Containing Term ID (List/Numpy Array)

        Outputs:
            decoded_entry_term     : Decoded Entry Term (String)
    """
    def Decode_CL_Input_Instance( self, encoded_input_instance, entry_term_mask = None ):
        # Check(s)
        if isinstance( encoded_input_instance, np.ndarray ): encoded_input_instance = list( encoded_input_instance )

        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Input_Instance() - Term ID: " + str( encoded_input_instance ) )

        if len( encoded_input_instance ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_CL_Input_Instance() - Error: Encoded Term Instance List Is Empty" )
            return ""

        decoded_entry_term = self.Get_Token_From_ID( encoded_input_instance[0] )

        if decoded_entry_term == -1:
            self.Print_Log( "BioCreativeDataLoader::Decode_CL_Input_Instance() - Error: Term ID Does Not Exist In Token Dictionary" )
            return ""

        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Input_Instance() - Decoding Output Label Instance: " + str( decoded_entry_term ) )

        return decoded_entry_term

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
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_labels, np.ndarray ) and encoded_output_labels.shape[0] == 0 or \
           isinstance( encoded_output_labels, COO        ) and encoded_output_labels.shape[0] == 0 or \
           isinstance( encoded_output_labels, csr_matrix ) and encoded_output_labels.shape[0] == 0:
            self.Print_Log( "BioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        # Convert Data-Types 'csr_matrix' And 'COO' Into Numpy Arrays
        if isinstance( encoded_output_labels, COO ):
            encoded_output_labels = np.asarray( encoded_output_labels.todense() )

        if isinstance( encoded_output_labels, csr_matrix ):
            encoded_output_labels = np.asarray( encoded_output_labels.todense() )

        # Preprocessing Of Numpy Array Data-Type
        if isinstance( encoded_output_labels, np.ndarray ):
            # Round Elements To Nearest Tenth
            encoded_output_labels = np.round( encoded_output_labels )

            # Convert Numpy Array To List
            encoded_output_labels = list( encoded_output_labels )

        decoded_output_labels = []

        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Output_Instance() - Decoding Output Label Instance: " + str( encoded_output_labels ) )

        for idx, val in enumerate( encoded_output_labels ):
            # Perform Thresholding
            #   Note - This Is Only Applicable When Utilizing 'Sigmoid' As The Final Activation Function.
            #          As 0.5 Is The Inflection Point Of The Function.
            if val > 0.5:
                # Fetch Decoded Concept ID String Using Index
                decoded_output = self.Get_Concept_From_ID( idx )

                # Append Concept ID String To Decoded Concept ID List
                if decoded_output not in decoded_output_labels: decoded_output_labels.append( decoded_output )

        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Output_Instance() - Decoded Output Label Instance: " + str( decoded_output_labels ) )

        return decoded_output_labels

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self, encoded_input_instance, entry_term_mask = None, encoded_output_labels = [] ):
        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Instance() - Encoded Sequence     : " + str( encoded_input_instance ) )
        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Instance() - Encoded Output Labels: " + str( encoded_output_labels ) )

        decoded_entry_term    = self.Decode_CL_Input_Instance( encoded_input_instance = encoded_input_instance )
        decoded_output_labels = self.Decode_CL_Output_Instance( encoded_output_labels = encoded_output_labels )

        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Instance() - Decoded Entry Term   : " + str( decoded_entry_term ) )
        self.Print_Log( "BioCreativeDataLoader::Decode_CL_Instance() - Decoded Output Labels: " + str( decoded_output_labels ) )

        return decoded_entry_term, decoded_output_labels

    """
        Reads Original Data File And Compares Original Passage (Text) Sequences To The Pre-Processed Counterparts For Each Document By
          Their Associated Document ID. Pre-Processed Instance Sequence Data Is Converted To The Un-Processed Form By Aligning The BERT
          Tokenizer Pre-Processed Sequence Data. This Also Keep Track Of Entity Token Labels And Exact Span Indices Of Each Token Per
          Entity Label.

        NOTE: This Function Is Dependent On The Evaluation File. It Reads All Documents/Passages While Mapping Each Passage Instance
              To The Pre-Processed Instance And Re-populating The Evaluation Document Passage Data To Create The Formatted Output File.

              This May Be Fixed To Not Require The Evaluation File In The Future.

        TODO: Finish Outputting BioC Format For NER and NER + CL Predictions.

        Inputs:
            read_file_path            : Original Evaluation File To Read From (String)
            write_file_path           : Write File Path / Path To Write BioC Formatted Output Data (String)
            data_list                 : List Of Passage Objects, Parsed From BioC XML/JSON File (List Of Passage Objects)
            encoded_ner_inputs        : List Of Tokenized Encoded NER Input Instances (List Of Token IDs)
            encoded_ner_outputs       : List Of Tokenized Encoded NER Output Instances (List Of Label IDs)
            encoded_concept_inputs    : List Of Encoded Concept Input Instances (List Of Token IDs)
            encoded_concept_outputs   : List Of Encoded Concept Output Instances (List Of Label IDs)

        Outputs:
            None
    """
    def Write_Formatted_File( self, read_file_path = "", write_file_path = "bioc_formatted_file.xml", data_list = [],
                              encoded_ner_inputs = None, encoded_ner_outputs = None, encoded_concept_inputs = None, encoded_concept_outputs = None ):
        # Check(s)
        if encoded_ner_inputs is None:
            self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded NER Inputs == None" )
        if encoded_ner_outputs is None:
            self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded NER Outputs == None" )
        if encoded_concept_inputs is None:
            self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded Concept Inputs == None" )
        if encoded_concept_outputs is None:
            self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded Concept Outputs == None" )

        # Concept Output Data-Type Check
        if isinstance( encoded_concept_outputs, list ): encoded_concept_outputs = np.asarray( encoded_concept_outputs )

        # Use Data Instances Passed By Parameter, If 'data_list = None' Use Internally Stored Instances
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Warning: 'data_list' Is Empty / Using Data List Stored In Memory" )
            data_list = self.data_list

        # Storage Lists For NER & CL
        decoded_ner_sequences, decoded_ner_terms, decoded_ner_labels, original_cl_sequences, decoded_cl_terms, decoded_cl_labels = [], [], [], [], [], []

        # Open BioC Read And Write File Handles
        if read_file_path == "": read_file_path = self.data_file_path
        reader = bioc.BioCXMLDocumentReader( read_file_path )
        writer = bioc.BioCXMLDocumentWriter( write_file_path )

        # Write Collection Info
        collection_info = reader.get_collection_info()
        writer.write_collection_info( collection_info )

        # Compose A Concept Index To Concept ID List
        concept_id_list = list( self.Get_Concept_ID_Dictionary().keys() )

        # Let's Capitalize All Concept ID Elemnets In The List
        concept_id_list = [ concept_id.upper() for concept_id in concept_id_list ]

        # Get List Of Document Identifiers From Each Instance
        document_identifier_list = [ passage.Get_Document_ID() for passage in data_list ]

        # Extract Sequence IDs If 'encoded_ner_inputs' And 'encoded_cl_inputs' Are Tuples
        if isinstance( encoded_ner_inputs, tuple ): encoded_ner_inputs = encoded_ner_inputs[0]

        # Process Each Document In The File
        for document in reader:
            # Get Indices Of Document ID
            document_passage_indices = []
            desired_cl_indices       = []
            index_position           = 0
            annotation_id            = 0

            # See If The Current Document ID Within The Read File Exists Within The Parsed Instances In Memory
            #  If So, Get Which Indices (Data List Instances) Correspond To The Current Document
            while True:
                try:
                    index_position = document_identifier_list.index( document.id, index_position )
                    document_passage_indices.append( index_position )
                    index_position += 1
                    if index_position >= len( document_identifier_list ): break
                except ValueError as e:
                    break

            # Proceed To Convert The Pre-Processed Sequences Of Text To Their Original Un-Processed Form
            if len( document_passage_indices ) > 0:
                # Decode NER Sequences, Terms And Labels
                #   Iterate Through NER Encoded Input & Output Instances And Decode Them
                # if encoded_ner_inputs is not None and encoded_ner_outputs is not None:
                #     decoded_ner_sequences = np.asarray( encoded_ner_inputs[document_passage_indices] )
                #     decoded_ner_labels    = np.asarray( encoded_ner_outputs[document_passage_indices] )

                # Decode CL Sequences, Terms And Labels
                #   Iterate Through Concept Linking Encoded Input & Output Instances And Decode Them
                if encoded_concept_inputs is not None and encoded_concept_outputs is not None:
                    # Get Instance Indices Corresponding To The Document Passage Indices / Find Indices Which Have Been Parsed Within The Encoding Function
                    desired_cl_indices    = [ idx for idx, instance_idx in enumerate( self.concept_instance_data_idx ) if instance_idx in document_passage_indices ]
                    original_cl_sequences = [ data_list[self.concept_instance_data_idx[idx]].Get_Passage_Original() for idx in desired_cl_indices ]
                    decoded_cl_terms      = list( self.Get_Token_From_ID( encoded_concept_inputs[idx] ) for idx in desired_cl_indices )
                    decoded_cl_labels     = [ [ concept_id_list[idx] for idx, val in enumerate( encoded_concept_outputs[desired_idx] ) if val == 1.0 ] for desired_idx in desired_cl_indices ]

                # Storage Lists For NER & CL
                decoded_ner_sequences, decoded_ner_labels = [], []

                # Decode NER Sequences And Labels
                #   Iterate Through NER Encoded Input & Output Instances And Decode Them
                # for encoded_ner_sequence_instance, encoded_ner_label_instance in zip( encoded_ner_sequences, encoded_ner_labels ):
                #     # Decode The Input/Output Instance Into A Sequence Of Text Tokens In Addition To The Entity Labels Per Token
                #     decoded_ner_sequence_instance, decoded_ner_label_instance = self.Decode_NER_Instance( encoded_ner_sequence_instance, encoded_ner_label_instance,
                #                                                                                           remove_padding = True, remove_special_characters = True )
                #     decoded_ner_sequences.append( decoded_ner_sequence_instance )
                #     decoded_ner_labels.append( decoded_ner_label_instance )

                # NER/CL - Iterate Through Each Passage In The Given Document
                for passage in document.passages:
                    # Clear The Previous Sequence Annotations
                    passage.annotations.clear()

                    # Check / Skip Current Passage If Not Found Within Our Original Annotations List
                    if passage.text not in original_cl_sequences: continue

                    passage_offset = int( passage.offset )

                    # Check For Sequences Containing No Data Or Only Containing Whitespace
                    if len( passage.text ) == 0 or len( passage.text.split() ) == 0: continue

                    # Write BioC Annotation Data For Both NER + CL
                    if len( decoded_ner_terms ) > 0 and len( decoded_cl_terms ) > 0:
                        pass

                    # Write BioC Annotation Data For NER
                    elif len( decoded_ner_terms ) > 0 and len( decoded_cl_terms ) == 0:
                        pass

                    # Write BioC Annotation Data For CL
                    elif len( decoded_ner_terms ) == 0 and len( decoded_cl_terms ) > 0:
                        instance_idx = 0

                        for cl_index, sequence, term, labels in zip( desired_cl_indices, original_cl_sequences, decoded_cl_terms, decoded_cl_labels ):
                            # Skip Sequence/Passage Is Not Our Desired Sequence
                            if sequence != passage.text:
                                instance_idx = 0
                                continue

                            # Fetch CL Passage Terms And Labels
                            passage_terms    = data_list[self.concept_instance_data_idx[cl_index]].Get_Annotations()
                            passage_labels   = data_list[self.concept_instance_data_idx[cl_index]].Get_Annotation_Labels()
                            passsage_indices = data_list[self.concept_instance_data_idx[cl_index]].Get_Annotation_Indices()

                            is_composite_instance = True if True in data_list[self.concept_instance_data_idx[cl_index]].Get_Composite_Mention_List() else False

                            # We Skip Composite Mentions For Now
                            #   Also Skip Instances Which Are Duplicates
                            if is_composite_instance or instance_idx >= len( passsage_indices ):
                                self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Error: Composite Or Duplicate Instance Detected" )
                                continue

                            indices              = passsage_indices[instance_idx].split( ":" )
                            annotation_start_idx = passage_offset + int( indices[0] )
                            annotation_length    = int( indices[1] ) - int( indices[0] )

                            # Create New BioC Annotation Object Instance And Store Appropriate Information
                            annotation           = BioCAnnotation()
                            annotation.id        = str( annotation_id )
                            annotation.text      = str( sequence[ int( indices[0] ) : int( indices[1] ) ] )
                            annotation.infons["type"] = passage_labels[instance_idx].upper()

                            # Format Concept ID Labels For Passage Instance
                            concept_id_labels = ",".join( labels ) if len( labels ) > 0 else "-"

                            # Assign The Concept ID Labels For The Given Annotation Instance (Entry Term)
                            annotation.infons["identifier"] = concept_id_labels

                            # Create New BioCLocation Object Instance For BioC Annotation Object Instance
                            #   And Store Location Information.
                            location = BioCLocation( offset = annotation_start_idx, length = annotation_length )
                            annotation.locations.append( location )
                            passage.annotations.append( annotation )

                            # Increment The Annotation ID & Instance Counter
                            annotation_id += 1
                            instance_idx  += 1

                writer.write_document( document )

        # Close BioC XML Writer
        writer.close()

        self.Print_Log( "BioCreativeDataLoader::Write_Formatted_File() - Complete" )
        return True


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    def Load_Token_ID_Key_Data( self, file_path ):
        if self.Get_Number_Of_Unique_Tokens() > 0 or self.Get_Number_Of_Unique_Concepts() > 0:
            self.Print_Log( "BioCreativeDataLoader::Load_Token_ID_Key_Data() - Warning: Primary Key Hash Is Not Empty / Saving Existing Data To: \"temp_primary_key_data.txt\"", force_print = True )
            self.Save_Token_ID_Key_Data( "temp_key_data.txt" )

        self.token_id_dictionary, self.concept_id_dictionary = {}, {}

        file_data = self.utils.Read_Data( file_path = file_path )

        if len( file_data ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Load_Token_ID_Key_Data() - Error Loading File Data: \"" + str( file_path ) + "\"" )
            return False

        self.Print_Log( "BioCreativeDataLoader::Load_Token_ID_Data() - Loading Key Data" )

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
            self.Print_Log( "BioCreativeDataLoader::Load_Token_ID_Data() - Key: " + str( key ) + " - Value: " + str( value ) )
            if token_flag:   self.token_id_dictionary[key]   = int( value )
            if concept_flag: self.concept_id_dictionary[key] = int( value )

        self.Print_Log( "BioCreativeDataLoader::Load_Token_ID_Data() - Complete" )

        return True

    def Save_Token_ID_Key_Data( self, file_path ):
        if len( self.token_id_dictionary ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Save_Token_ID_Key_Data() - Warning: Primary Key Data = Empty / No Data To Save" )
            return

        self.Print_Log( "BioCreativeDataLoader::Save_Token_ID_Data() - Saving Key Data" )

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

        self.Print_Log( "BioCreativeDataLoader::Save_Token_ID_Data() - Complete" )

    """
        Generates IDs For Each Token Given The Following File Format

            Expected Format:    Seqences Of Text

        Inputs:
            data_list                    : List Of 'Passage' Class Objects
            lowercase                    : True = Cased Text, False = Uncased Text (Bool)
            scale_embedding_weight_value : Scales Embedding Weights By Specified Value ie. embedding_weights *= scale_embedding_weight_value (Float)
            update_dict                  : Forces Function To Run, Updating The Token ID Dictionary (If Not Using Embeddings) (Bool)

        Outputs:
            None
    """
    def Generate_Token_IDs( self, data_list = [], lowercase = False, scale_embedding_weight_value = 1.0, update_dict = False, concept_delimiter = "," ):
        # Check(s)
        if self.generated_embedding_ids and update_dict == False:
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Warning: Already Generated Embedding Token IDs" )
            return

        if len( data_list ) > 0 and self.Get_Number_Of_Embeddings_A() > 0 and self.generated_embedding_ids == False:
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Warning: Token IDs Cannot Be Generated From Data List When Embeddings Have Been Loaded In Memory" )
            return

        if update_dict:
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Updating Token ID Dictionary" )

        # Check(s)
        # If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        if self.Get_Number_Of_Embeddings_A() > 0:
            # Only Generate Token ID Dictionary Using Embeddings Once.
            #   This Is Skipped During Subsequent Calls To This Function
            if not self.generated_embedding_ids:
                # Insert Padding At First Index Of The Token ID Dictionary
                padding_token = self.padding_token.lower() if lowercase else self.padding_token
                self.token_id_dictionary[padding_token] = 0

                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( self.Get_Number_Of_Embeddings_A() + 1, len( self.embeddings_a[1].split() ) - 1 ) )

                self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Generating Token IDs Using Embeddings" )

                # Parse Embeddings
                for index, embedding in enumerate( self.embeddings_a, 1 ):
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
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
                        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( embedding_text ) + "\" => Embedding Row Index: " + str( index ) )
                        self.token_id_dictionary[embedding_text] = index
                    else:
                        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Token - Warning: \"" + str( embedding_text ) + "\" Already In Dictionary" )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_input_tokens  = len( self.token_id_dictionary )
                self.number_of_output_tokens = len( self.annotation_labels   )

                self.embeddings_a = []
                self.embeddings_a = np.asarray( embeddings ) * scale_embedding_weight_value if scale_embedding_weight_value != 1.0 else np.asarray( embeddings )

                self.embeddings_a_loaded     = True
                self.generated_embedding_ids = True

        # Generate One-Hot Encoding Using Data
        else:
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Generating Token IDs Using Data" )
            self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Processing Data List Elements" )

            # Insert Padding Token At First Index Of The Token ID Dictionaries
            padding_token = self.padding_token.lower() if lowercase else self.padding_token
            if padding_token not in self.token_id_dictionary:
                self.token_id_dictionary[self.padding_token] = self.number_of_input_tokens
                self.number_of_input_tokens += 1

            # Process Sequences In Data List
            for passage in data_list:
                self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Processing Sequence: " + str( passage.Get_Passage() ) )

                # Add Sequence Tokens To Token List
                tokens = passage.Get_Passage().split()

                # Add Concept Linking Entry Terms To Token List
                tokens += passage.Get_Annotations()

                # Check To See If Sequence Tokens Are Already In Dictionary, If Not Add The Tokens
                for token in tokens:
                    if lowercase: token = token.lower()
                    if token not in self.token_id_dictionary:
                        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( token ) + "\" Value: " + str( self.number_of_input_tokens ) )
                        self.token_id_dictionary[token] = self.number_of_input_tokens
                        self.number_of_input_tokens += 1
                    else:
                        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Token - Warning: \"" + str( token ) + "\" Already In Dictionary" )

            # Set Number Of Primary Tokens Based On Token ID Dictionary Length
            self.number_of_input_tokens  = len( self.token_id_dictionary )
            self.number_of_output_tokens = len( self.annotation_labels   )

        ###########################################
        # Concept Linking Unique Token Generation #
        ###########################################

        # Build Concept To Concept ID Dictionary
        cui_less_token = self.cui_less_token.lower() if lowercase else self.cui_less_token
        if cui_less_token not in self.concept_id_dictionary:
            self.concept_id_dictionary[cui_less_token] = self.number_of_concept_tokens
            self.number_of_concept_tokens += 1

        for passage in data_list:
            concepts = []

            # Include Annotation Entity Linking MESH ID Concept Tokens
            #   Let's Split All One-To-Many (Term-To-Concept) MESH IDs To Single MESH IDs
            for concept in passage.Get_Annotation_Concept_IDs():
                concepts.extend( concept.split( concept_delimiter ) ) if concept_delimiter in concept else concepts.append( concept )

            # Include Chemical Indexing MESH ID Concept Tokens
            #   We're Doing This To Include Potential OOV Concept Tokens
            concepts += passage.Get_Concepts().values()

            # Add All Tokens To Unique Concept Token Dictionary
            for concept in concepts:
                if lowercase: concept = concept.lower()
                if concept not in self.concept_id_dictionary:
                    self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Concept Token: \"" + str( concept ) + "\" Value: " + str( self.number_of_concept_tokens ) )
                    self.concept_id_dictionary[concept] = self.number_of_concept_tokens
                    self.number_of_concept_tokens += 1
                else:
                    self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Adding Concept - Warning: \"" + str( concept ) + "\" Already In Dictionary" )

        self.Print_Log( "BioCreativeDataLoader::Generate_Token_IDs() - Complete" )

    """
        Called When Loading Model
    """
    def Update_Token_IDs( self, data_list = [], lowercase = False ):
        pass

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)
            file_name : File Name (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path, file_name = "" ):
        self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Load Necessary Data-Loader Specific Data
        cfg_file_name = "cfg" if file_name == "" else file_name + "_cfg"
        data_loader_settings = self.utils.Read_Data( file_path = file_path + cfg_file_name )

        for setting_and_value in data_loader_settings:
            elements = setting_and_value.split( "<:>" )
            if elements[0] == "Max_Sequence_Length" : self.Set_Max_Sequence_Length( int( elements[-1] ) )
            if elements[0] == "Number_Of_Inputs"    : self.number_of_input_tokens  = int( elements[-1] )
            if elements[0] == "Number_Of_Outputs"   : self.number_of_output_tokens = int( elements[-1] )

        # Load Token Key ID File
        key_file_name = "key" if file_name == "" else file_name + "_key"
        self.Load_Token_ID_Key_Data( file_path + key_file_name )

        # Load Input/Output Matrices
        input_file_name, output_file_name = "", ""

        if len( file_name ) > 0:
            input_file_name  = str( file_name ) + "_"
            output_file_name = str( file_name ) + "_"

        # Input Matrix
        self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading Input Matrix" )

        if self.utils.Check_If_File_Exists( file_path + input_file_name +  "encoded_input.coo.npz" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading COO Format Input Matrix" )
            self.ner_inputs = load_npz( file_path + input_file_name +  "encoded_input.coo.npz" )
        elif self.utils.Check_If_File_Exists( file_path + input_file_name + "encoded_input.npz" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Input Matrix" )
            self.ner_inputs = scipy.sparse.load_npz( file_path + input_file_name +  "encoded_input.npz" )
        elif self.utils.Check_If_File_Exists( file_path + input_file_name +  "encoded_input.npy" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Input Matrix" )
            self.ner_inputs = np.load( file_path + input_file_name +  "encoded_input.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Input Matrix Not Found" )

        # Output Matrix
        self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading Output Matrix" )

        if self.utils.Check_If_File_Exists( file_path + output_file_name + "encoded_output.coo.npz" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading COO Format Output Matrix" )
            self.ner_outputs = load_npz( file_path + output_file_name + "encoded_output.coo.npz" )
        elif self.utils.Check_If_File_Exists( file_path + output_file_name + "encoded_output.npz" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Output Matrix" )
            self.ner_outputs = scipy.sparse.load_npz( file_path + output_file_name + "encoded_output.npz" )
        elif self.utils.Check_If_File_Exists( file_path + output_file_name + "encoded_output.npy" ):
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Output Matrix" )
            self.ner_outputs = np.load( file_path + output_file_name + "encoded_output.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Output Matrix Not Found" )

        self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Complete" )

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
        self.Print_Log( "BioCreativeDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

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
        self.Print_Log( "BioCreativeDataLoader::Save_Vectorized_Model_Data() - Saving Input Matrix" )

        if self.ner_inputs is None:
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Warning: Primary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.ner_inputs, COO ):
            save_npz( file_path + input_file_name + "encoded_input.coo.npz", self.ner_inputs )
        elif isinstance( self.ner_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + input_file_name + "encoded_input.npz", self.ner_inputs )
        elif isinstance( self.ner_inputs, np.ndarray ):
            np.save( file_path + input_file_name + "encoded_input.npy", self.ner_inputs, allow_pickle = True )

        # Output Matrix
        self.Print_Log( "BioCreativeDataLoader::Save_Vectorized_Model_Data() - Saving Output Matrix" )

        if self.ner_outputs is None:
            self.Print_Log( "BioCreativeDataLoader::Load_Vectorized_Model_Data() - Warning: Output == 'None' / No Data To Save" )
        elif isinstance( self.ner_outputs, COO ):
            save_npz( file_path + output_file_name + "encoded_output.coo.npz", self.ner_outputs )
        elif isinstance( self.ner_outputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + output_file_name + "encoded_output.npz", self.ner_outputs )
        elif isinstance( self.ner_outputs, np.ndarray ):
            np.save( file_path + output_file_name + "encoded_output.npy", self.ner_outputs, allow_pickle = True )

        self.Print_Log( "BioCreativeDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Fetches NER Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        self.Print_Log( "BioCreativeDataLoader::Get_Token_ID() - Fetching ID For Token: \"" + str( token ) + "\"" )

        if token in self.token_id_dictionary:
            if self.lowercase_text: token = token.lower()
            self.Print_Log( "BioCreativeDataLoader::Get_Token_ID() - Token ID Found: \"" + str( token ) + "\" => " + str( self.token_id_dictionary[token] ) )
            return self.token_id_dictionary[token]
        else:
            self.Print_Log( "BioCreativeDataLoader::Get_Token_ID() - Unable To Locate Token In Dictionary" )

        self.Print_Log( "BioCreativeDataLoader::Get_Token_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches Concept Token ID From String.

        Inputs:
            concept    : Token (String)

        Outputs:
            concept_id : Token ID Value (Integer)
    """
    def Get_Concept_ID( self, concept ):
        self.Print_Log( "BioCreativeDataLoader::Get_Concept_ID() - Fetching ID For Concept: \"" + str( concept ) + "\"" )

        if self.lowercase_text: concept = concept.lower()

        if concept in self.concept_id_dictionary:
            self.Print_Log( "BioCreativeDataLoader::Get_Concept_ID() - Token ID Found: \"" + str( concept ) + "\" => " + str( self.concept_id_dictionary[concept] ) )
            return self.concept_id_dictionary[concept]
        else:
            self.Print_Log( "BioCreativeDataLoader::Get_Concept_ID() - Unable To Locate Concept In Dictionary" )

        self.Print_Log( "BioCreativeDataLoader::Get_Concept_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches NER Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Token String (String)
    """
    def Get_Token_From_ID( self, index_value ):
        self.Print_Log( "BioCreativeDataLoader::Get_Token_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.token_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "BioCreativeDataLoader::Get_Token_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "BioCreativeDataLoader::Get_Token_From_ID() - Warning: Key Not Found In Dictionary" )

        return None

    """
        Fetches Concept Token String From ID Value.

        Inputs:
            index_value  : Concept ID Value (Integer)

        Outputs:
            key          : Concept TOken String (String)
    """
    def Get_Concept_From_ID( self, index_value ):
        self.Print_Log( "BioCreativeDataLoader::Get_Concept_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.concept_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "BioCreativeDataLoader::Get_Concept_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "BioCreativeDataLoader::Get_Concept_From_ID() - Warning: Key Not Found In Dictionary" )

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
    #    Worker Thread Function                                                                #
    #                                                                                          #
    ############################################################################################

    """
        DataLoader Model Data Vectorization Worker Thread

        Inputs:
            thread_id              : Thread Identification Number (Integer)
            data_list              : List Of String Instances To Vectorize (Data Chunk Determined By BioCreativeDataLoader::Encode_NER_Model_Data() Function)
            dest_array             : Placeholder For Threaded Function To Store Outputs (Do Not Modify) (List)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays

        Outputs:
            inputs                 : CSR Matrix or Numpy Array
            outputs                : CSR Matrix or Numpy Array

        Note:
            Outputs Are Stored In A List Per Thread Which Is Managed By BioCreativeDataLoader::Encode_NER_Model_Data() Function.

    """
    def Worker_Thread_Function( self, thread_id, data_list, dest_array, use_csr_format = False ):
        self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Vectorizing Data Using Settings" )
        self.Print_Log( "                                                - Thread ID: " + str( thread_id ) + " - Use CSR Format    : " + str( use_csr_format ) )

        # Vectorized Input/Output Placeholder Lists
        input_sequences    = []
        annotation_mapping = []
        concept_mapping    = []

        # CSR Matrix Format
        if use_csr_format:
            input_row_index  = 0
            output_row_index = 0

            input_row,  output_row  = [], []
            input_col,  output_col  = [], []
            input_data, output_data = [], []
            output_depth_idx        = []

        # Iterate Through List Of 'Passage' Class Objects
        for passage in data_list:
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Text Sequence: " + str( passage.Get_Passage().rstrip() ) )

            # Check
            if not passage.Get_Passage() or len( passage.Get_Passage() ) <= 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Text Contains No Data" )
                continue

            if not passage.Get_Annotations() or len( passage.Get_Annotations() ) == 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Annotation List Contains No Data" )
                continue

            if not passage.Get_Annotation_Labels() or len( passage.Get_Annotation_Labels() ) == 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Annotation Label List Contains No Data" )
                continue

            encoded_sequence, encoded_annotation_sequence = self.Encode_NER_Instance( passage.Get_Passage().rstrip(), passage.Get_Annotations(), passage.Get_Annotation_Labels(),
                                                                                      passage.Get_Annotation_Indices(), composite_mention_list = passage.Get_Composite_Mention_List(),
                                                                                      individual_mention_list = passage.Get_Individual_Mention_List() )

            # Check(s)
            if len( encoded_sequence ) == 0 or len( encoded_annotation_sequence ) == 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Error Occurred While Encoding Text Sequence", force_print = True )
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() -           Text Sequence      : '" + str( passage.Get_Passage().rstrip()  ) + "'", force_print = True )
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() -           Annotations        : '" + str( passage.Get_Annotations()       ) + "'", force_print = True )
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() -           Annotations Labels : '" + str( passage.Get_Annotation_Labels() ) + "'", force_print = True )
                continue

            ############################
            # Input & Output Sequences #
            ############################
            if use_csr_format:
                # Input
                for i, value in enumerate( encoded_sequence ):
                    if value == 0: break

                    input_row.append( input_row_index )
                    input_col.append( i )
                    input_data.append( value )

                # Output
                for i, token_instance in enumerate( encoded_annotation_sequence ):
                    for tag_index in range( len( token_instance ) ):
                        if encoded_annotation_sequence[i][tag_index] > 0:
                            output_row.append( output_row_index )
                            output_col.append( i )
                            output_depth_idx.append( tag_index )
                            output_data.append( 1 )

                input_row_index  += 1
                output_row_index += 1
            else:
                input_sequences.append( encoded_sequence )
                annotation_mapping.append( encoded_annotation_sequence )

        # Set Variable Lengths For Input/Output Data Matrices/Lists
        # Assumes Inputs Are Embeddings Indices (Length = 1) And Output Is Binary Classification (Length = 1)
        number_of_input_rows  = self.max_sequence_length
        number_of_output_rows = self.max_sequence_length
        output_label_dim      = len( self.annotation_labels )

        # Convert Inputs & Outputs To CSR Matrix Format
        if use_csr_format:
            # Convert Lists To Numpy Arrays
            input_row  = np.asarray( input_row  )
            input_col  = np.asarray( input_col  )
            input_data = np.asarray( input_data )

            # Check(s)
            if len( input_data ) == 0 and len( output_data ) == 0:
                dest_array[thread_id] = None
                return

            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Data :" )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Row Size                : "  + str( len( input_row     ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Col Size                : "  + str( len( input_col     ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Data Size               : "  + str( len( input_data    ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Number Of Matrix Columns: "  + str( input_row_index      ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Number Of Matrix Rows   : "  + str( number_of_input_rows ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Matrix Row              :\n" + str( input_row            ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Matrix Column           :\n" + str( input_col            ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Matrix Data             :\n" + str( input_data           ) )

            # Check(s)
            if ( len( input_row ) == 0 or len( input_col ) == 0 or len( input_data ) == 0 ) and input_row_index == 0 or number_of_input_rows == 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Input Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            # Convert Numpy Arrays To CSR Matrices
            input_sequences  = csr_matrix( ( input_data, ( input_row, input_col ) ), shape = ( input_row_index, number_of_input_rows ) )

            # Convert Outputs To CSR Matrix Format
            output_row       = np.asarray( output_row       )
            output_col       = np.asarray( output_col       )
            output_depth_idx = np.asarray( output_depth_idx )
            output_data      = np.asarray( output_data      )

            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data :" )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Row Size                : "  + str( len( output_row     ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Col Size                : "  + str( len( output_col     ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data Size               : "  + str( len( output_data    ) ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Number Of Matrix Columns: "  + str( output_row_index      ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Number Of Matrix Rows   : "  + str( number_of_output_rows ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Row              :\n" + str( output_row            ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Column           :\n" + str( output_col            ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Matrix Data             :\n" + str( output_data           ) )

            # Check(s)
            if len( output_row ) == 0 or len( output_col ) == 0 or len( output_depth_idx ) == 0 or len( output_data ) == 0 and output_row_index == 0 or number_of_output_rows == 0:
                self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Error: Output Matrix Dimensions Do Not Match" )
                dest_array[thread_id] = None
                return

            # Convert Numpy Arrays To CSR Matrices
            annotation_mapping = COO( [ output_row, output_col, output_depth_idx ], output_data, shape = ( output_row_index, number_of_output_rows, output_label_dim ), fill_value = self.annotation_labels["O"] )
        else:
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Data :\n" + str( input_sequences    ) )
            self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data:\n" + str( annotation_mapping ) )

            input_sequences    = np.asarray( input_sequences )
            annotation_mapping = np.asarray( annotation_mapping )

            # Check(s)
            if len( input_sequences ) == 0:
                dest_array[thread_id] = None
                return

        # Assign Thread Vectorized Data To Temporary DataLoader Placeholder Array
        dest_array[thread_id] = [input_sequences, annotation_mapping]

        self.Print_Log( "BioCreativeDataLoader::Worker_Thread_Function() - Complete" )



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NERLink.DataLoader import BioCreativeDataLoader\n" )
    print( "     data_loader = BioCreativeDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

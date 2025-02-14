#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/06/2022                                                                   #
#    Revised: 11/12/2022                                                                   #
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

# Suppress Warnings/FutureWarnings
import warnings
warnings.filterwarnings( 'ignore' )
#warnings.simplefilter( action = 'ignore', category = Warning )
#warnings.simplefilter( action = 'ignore', category = FutureWarning )   # Also Works For Future Warnings

# Standard Modules
import os

# Suppress Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes Tensorflow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import re
import numpy as np

# Custom Modules
from NERLink.DataLoader.BioCreative import BERTBioCreativeDataLoader


############################################################################################
#                                                                                          #
#   BERT Embedding Similarity Model Data Loader Model Class                                #
#                                                                                          #
############################################################################################

class BERTSimilarityDataLoader( BERTBioCreativeDataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  bert_model = "bert-base-cased", skip_individual_mentions = False, skip_composite_mentions = False, lowercase = False, ignore_label_type_list = [] ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle, lowercase = lowercase,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          skip_individual_mentions = skip_individual_mentions, skip_composite_mentions = skip_composite_mentions,
                          ignore_label_type_list = ignore_label_type_list, bert_model = bert_model )
        self.version = 0.02

    """
        Vectorized/Binarized NER Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list          : List Of Passage Objects
            use_csr_format     : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            keep_in_memory     : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            number_of_threads  : Number Of Threads To Deploy For Data Vectorization (Integer)
            is_validation_data : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
    """
    def Encode_NER_Model_Data( self, data_list = [], use_csr_format = False, keep_in_memory = True, number_of_threads = 4, is_validation_data = False, is_evaluation_data = False ):
        raise NotImplementedError

    """
        Encodes Concept Linking Data - Used For Training, Validation Or Evaluation Data

        Inputs:
            data_list            : List Of Passage Objects
            use_csr_format       : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            keep_in_memory       : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            is_validation_data   : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data   : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
            pad_input            : Adds Padding To Input Sequence ie. [PAD] Tokens After Actual Sequence Until Max Sequence Length (Bool)
            pad_output           : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                   Categorical Crossentropy vs. Sparse Categorical Crossentropy
            concept_delimiter    : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                   Used For One-To-Many Relationships
            mask_term_sequence   : True  -> Entry Term Mask = Entire Sub-Word Sequence Containing Entry Term
                                   False -> Encode Just Entry Term Sub-Word Tokens
            separate_sentences   : Separates Sentences With [SEP] Token Using Sentence Delimiters (Bool)
            term_sequence_only   : Disregards All Sequences Surrounding The Sequence Of Interest, Only Encoding The Desired Sequence (Bool)
            restrict_context     : Restricts Or Reduces The Sequence Context Surrounding The Entry Term Used To Generate Its Embedding (Bool)
            label_per_sub_word   : Produces An Output Label For Each Sub-Word Token In The Sequence (Bool)

        Outputs:
            concept_inputs       : Tuple Of Encoded Concept Model Inputs
                                         1) Encoded Token IDs
                                         2) Attention Masks
                                         3) Token Type IDs
                                         4) Entry Term Masks
            concept_outputs      : Encoded Concept Labels (CSR, COO Matrix or Numpy Array)
    """
    def Encode_CL_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = True, keep_in_memory = True,
                              is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                              mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = True,
                              use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        # BERT Data Loader Does Not Support CSR/COO Formats At The Moment
        if use_csr_format and label_per_sub_word:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_NER_Model_Data() - Use CSR Format Not Supported When 'label_per_sub_word = True' / Setting 'use_csr_format = False'" )
            use_csr_format = False

        # Clear Previous Concept Instance Data Index List
        self.concept_instance_data_idx.clear()

        # Enforce BERT Max Sequence Length Limitation
        if self.max_sequence_length > self.max_sequence_limit:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Warning: Max Sequence Length > " + str( self.max_sequence_limit ) + " / Enforcing BERT Max Sequence Length == "  + str( self.max_sequence_limit ) )
            self.max_sequence_length = self.max_sequence_limit

        self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Encoding Concept Instances" )

        encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks, encoded_concept_outputs = [], [], [], [], []

        for passage_idx, passage in enumerate( data_list ):
            # Encode All Term and Concept Pairs
            for annotation_tokens, annotation_idx, annotation_concepts, is_composite_mention, is_individual_mention in zip( passage.Get_Annotations(), passage.Get_Annotation_Indices(),
                                                                                                                            passage.Get_Annotation_Concept_IDs(), passage.Get_Composite_Mention_List(),
                                                                                                                            passage.Get_Individual_Mention_List() ):
                if annotation_tokens == "" or annotation_concepts == "":
                    self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Warning: Instance Contains No Entry Terms Or Concept IDs" )
                    self.Print_Log( "                                                       - Sequence: " + str( passage.Get_Passage() ) )
                    continue
                elif is_composite_mention and self.skip_composite_mentions:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Composite Mention Detected / Skipping Composite Mention" )
                    continue
                elif is_individual_mention and self.skip_individual_mentions:
                    self.Print_Log( "BioCreativeDataLoader::Encode_CL_Model_Data() - Individual Mention Detected / Skipping Individual Mention" )
                    continue

                # Tokenize The Sequence Using The BERT Tokenizer And Extract The Tokenized Sequence Information (IDs Per Concept Entry-Term Sub-Word)
                encoded_input_instance, encoded_output_instance = self.Encode_CL_Instance( text_sequence = passage.Get_Passage(), entry_term = annotation_tokens,
                                                                                           annotation_concept = annotation_concepts, annotation_indices = annotation_idx,
                                                                                           pad_input = pad_input, pad_output = pad_output, term_sequence_only = term_sequence_only,
                                                                                           concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                                                           separate_sentences = separate_sentences, restrict_context = restrict_context,
                                                                                           label_per_sub_word = label_per_sub_word )

                if encoded_input_instance is None or encoded_output_instance is None:
                    self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Input And/Or Output Instance" )
                    continue

                # Extract Encoded Input Elements From 'encoded_input_instance' Tuple
                token_ids, attention_mask, token_type_ids, entry_term_mask = encoded_input_instance

                # Store Encoded Instance Elements In Their Appropriate Lists
                encoded_token_ids.append( token_ids )
                encoded_attention_masks.append( attention_mask )
                encoded_token_type_ids.append( token_type_ids )
                encoded_entry_term_masks.append( entry_term_mask )

                # Concept Output Is A Vector/Array Of 'N' Classes With Our Desired Instance Class As '1'
                encoded_concept_outputs.append( encoded_output_instance )

                # Keep Track Of Which Passage The Instance Came From
                self.concept_instance_data_idx.append( passage_idx )

        # Convert Data To Numpy Arrays
        encoded_token_ids        = np.asarray( encoded_token_ids,        dtype = np.int32 )
        encoded_attention_masks  = np.asarray( encoded_attention_masks,  dtype = np.int32 )
        encoded_token_type_ids   = np.asarray( encoded_token_type_ids,   dtype = np.int32 )
        encoded_entry_term_masks = np.asarray( encoded_entry_term_masks, dtype = np.int32 )
        encoded_concept_outputs  = np.asarray( encoded_concept_outputs,  dtype = np.float32 )

        # Check(s)
        number_of_input_instances  = encoded_token_ids.shape[0]
        number_of_output_instances = encoded_concept_outputs.shape[0]

        if number_of_input_instances == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Inputs", force_print = True )
            return None, None
        elif number_of_output_instances == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Outputs", force_print = True )
            return None, None
        elif number_of_input_instances != number_of_output_instances:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Error: Number Of Input And Output Instances Not Equal", force_print = True )
            return None, None

        if keep_in_memory:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Storing Encoded Data In Memory" )

            if is_validation_data:
                self.concept_val_inputs   = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_val_outputs  = encoded_concept_outputs
            elif is_evaluation_data:
                self.concept_eval_inputs  = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_eval_outputs = encoded_concept_outputs
            else:
                self.concept_inputs       = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_outputs      = encoded_concept_outputs

        self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Model_Data() - Complete" )

        return ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks ), encoded_concept_outputs

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            text_sequence            : Sequence Of Text
    """
    def Encode_NER_Instance( self, text_sequence, annotations, annotation_labels, annotation_indices, composite_mention_list = [], individual_mention_list = [], use_padding = True ):
        raise NotImplementedError

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            annotation_concept   : Concept ID / CUI (String)
            pad_output           : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                   i.e. Categorical Crossentropy Loss vs. Sparse Categorical Crossentropy Loss Formats
            concept_delimiter    : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                   Used For One-To-Many Relationships
            label_per_sub_word   : Produces An Output Label For Each Sub-Word Token In The Sequence (Bool)
            encoded_inputs       : Encoded Inputs - Returned Back From self.Encoded_CL_Input_Instance() Function (Not Used)

        Outputs:
            encoded_concept      : Candidate Concept Embedding
    """
    def Encode_CL_Output_Instance( self, annotation_concept, pad_output = False, concept_delimiter = None, label_per_sub_word = False, encoded_inputs = None ):
        # Check(s)
        if len( self.concept_id_dictionary ) == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None

        self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Encoding Concept: " + str( annotation_concept ) )

        encoded_concept = []

        if concept_delimiter is not None:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Computing Average Of Concept(s)" )

            for concept_idx, concept in enumerate( annotation_concept.split( concept_delimiter ) ):
                # Skip Composite Mentions If 'self.skip_composite_mentions == True'
                if self.skip_composite_mentions and concept_idx > 0:
                    continue

                concept_id = self.Get_Concept_ID( concept )

                if concept_id == -1:
                    self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( concept ) + "\' Not In Dictionary", force_print = True )
                    return None

                # Fetch Concept Embedding
                encoded_concept.append( self.Get_Embeddings_B()[ concept_id ] )

            # Check To See If Any Concept IDs Were Encoded
            if len( encoded_concept ) == 0:
                self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Error: No Concept IDs Were Encoded For Multi-Output Instance", force_print = True )
                return None

            # Compute Average Of Concept Embeddings
            if len( encoded_concept ) > 1:
                encoded_concept = sum( encoded_concept ) / len( encoded_concept )
            else:
                encoded_concept = encoded_concept[0]
        else:
            concept_id = self.Get_Concept_ID( annotation_concept )

            if concept_id == -1:
                self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( annotation_concept ) + "\' Not In Dictionary", force_print = True )
                return None

            # Fetch Concept Embedding
            encoded_concept.append( self.Get_Embeddings_B()[ concept_id ] )

        self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Encoded Concept: " + str( encoded_concept ) )

        return encoded_concept

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            text_sequence        : Text Sequence In Which The 'entry_term' Occurs. (String)
            entry_term           : Concept Token (String)
            annotation_concept   : Concept ID / CUI (String)
            annotation_indices   : Concept Token Indices (String Of Two Integers Separated By ':' Character)
            pad_input            : Adds Padding To Input Sequence ie. [PAD] Tokens After Actual Sequence Until Max Sequence Length (Bool)
            pad_output           : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                   i.e. Categorical Crossentropy Loss vs. Sparse Categorical Crossentropy Loss Formats
            concept_delimiter    : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                   Used For One-To-Many Relationships
            mask_term_sequence   : True  -> Entry Term Mask = Entire Sub-Word Sequence Containing Entry Term
                                   False -> Encode Just Entry Term Sub-Word Tokens
            separate_sentences   : Separates Sentences With [SEP] Token Using Sentence Delimiters (Bool)
            term_sequence_only   : Disregards All Sequences Surrounding The Sequence Of Interest, Only Encoding The Desired Sequence (Bool)
            restrict_context     : Restricts Or Reduces The Sequence Context Surrounding The Entry Term Used To Generate Its Embedding (Bool)
            label_per_sub_word   : Produces An Output Label For Each Sub-Word Token In The Sequence (Bool)

        Outputs:
            encoded_entry_term   : Tuple Consisting Of:
                                         1) Encoded Text Sequence Sub-Word IDs
                                         2) Attention Masks
                                         3) Token Type IDs
                                         4) Entry Term Mask
            encoded_concept      : Candidate Concept Embedding
    """
    def Encode_CL_Instance( self, text_sequence, entry_term, annotation_concept, annotation_indices, pad_input = True, pad_output = False,
                            concept_delimiter = None, mask_term_sequence = False, separate_sentences = True, term_sequence_only = False,
                            restrict_context = False, label_per_sub_word = False ):
        # Check(s)
        if len( self.concept_id_dictionary ) == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None, None

        encoded_inputs = self.Encode_CL_Input_Instance( text_sequence = text_sequence, entry_term = entry_term, annotation_indices = annotation_indices,
                                                        pad_input = pad_input, mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                        term_sequence_only = term_sequence_only, restrict_context = restrict_context )
        if encoded_inputs is None:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Instance() - Error Encoding Input" )
            return None, None

        encoded_ouput  = self.Encode_CL_Output_Instance( annotation_concept = annotation_concept, pad_output = pad_output, concept_delimiter = concept_delimiter,
                                                         label_per_sub_word = label_per_sub_word, encoded_inputs = encoded_inputs )

        if encoded_ouput is None:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Instance() - Error Encoding Output" )
            return None, None

        return encoded_inputs, encoded_ouput
    """
        Decodes Sequence Of Token IDs to Sequence Of Token Strings

        Inputs:
            encoded_input_sequence : Sequence Of Token IDs
            remove_padding         : Removes Padding Tokens From Returned String Sequence
    """
    def Decode_NER_Input_Instance( self, encoded_input_sequence, remove_padding = False, convert_subwords_to_tokens = False ):
        raise NotImplementedError

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings

        Inputs:
            encoded_output_sequence : Sequence Of Encoded Label IDs
    """
    def Decode_NER_Output_Instance( self, encoded_output_sequence ):
        raise NotImplementedError

    """
        Decodes Input & Output Sequence Of NER Token IDs And Label IDs To Sequence Of NER Token & Label Strings

        Inputs:
            encoded_input_sequence    : Sequence Of Encoded Token IDs
            encoded_output_sequence   : Sequence Of Encoded Label IDs
            remove_padding            : Removes [PAD] Token From Decoded Input Sequence (True/False)
            remove_special_characters : Removed [CLS] and [SEP] Tokens From Decoded Input/Output Sequence (True/False)
    """
    def Decode_NER_Instance( self, encoded_input_sequence, encoded_output_sequence, remove_padding = True, remove_special_characters = False ):
        raise NotImplementedError

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
            self.Print_Log( "BERTSimilarityDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_labels, np.ndarray ) and encoded_output_labels.shape[0] == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Output_Instance() - Decoding Output Embedding: " + str( encoded_output_labels ) )

        decoded_output_labels = None

        for idx, embedding in enumerate( self.Get_Embeddings_B() ):
            embedding = np.asarray( embedding, dtype = np.float32 )
            if embedding == encoded_output_labels:
                decoded_output_labels = self.Get_Concept_From_ID( idx )

        if not decoded_output_labels: decoded_output_labels = "N/A"

        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Output_Instance() - Decoded Output Label Instance: " + str( decoded_output_labels ) )

        return decoded_output_labels

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self, encoded_input_instance, entry_term_mask, encoded_output_labels ):
        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Instance() - Encoded Sequence     : " + str( encoded_input_instance ) )
        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Instance() - Entry Term Mask      : " + str( entry_term_mask ) )
        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Instance() - Encoded Output Labels: " + str( encoded_output_labels ) )

        decoded_entry_term    = self.Decode_CL_Input_Instance( encoded_input_instance = encoded_input_instance, entry_term_mask = entry_term_mask )
        decoded_output_labels = self.Decode_CL_Output_Instance( encoded_output_labels = encoded_output_labels )

        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Instance() - Decoded Entry Term   : " + str( decoded_entry_term ) )
        self.Print_Log( "BERTSimilarityDataLoader::Decode_CL_Instance() - Decoded Output Labels: " + str( decoded_output_labels ) )

        return decoded_entry_term, decoded_output_labels


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

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
            self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Warning: Already Generated Embedding Token IDs" )
            return

        if len( data_list ) > 0 and len( self.embeddings_a ) > 0 and self.generated_embedding_ids == False:
            self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Warning: Token IDs Cannot Be Generated From Data List When Embeddings Have Been Loaded In Memory" )
            return

        if update_dict:
            self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Updating Token ID Dictionary" )

        # Check(s)
        # If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        # Insert Padding At First Index Of The Token ID Dictionary
        padding_token = self.padding_token.lower()  if lowercase else self.padding_token
        if padding_token not in self.token_id_dictionary:
            self.token_id_dictionary[padding_token] = self.number_of_input_tokens
            self.number_of_input_tokens += 1

        # Insert CUI-LESS Token At Second Index Of The Concept ID Dictionary
        cui_less_token = self.cui_less_token.lower() if lowercase else self.cui_less_token
        if cui_less_token not in self.concept_id_dictionary:
            self.concept_id_dictionary[cui_less_token] = self.number_of_concept_tokens
            self.number_of_concept_tokens += 1

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        # Only Generate Token ID Dictionary Using Embeddings Once.
        #   This Is Skipped During Subsequent Calls To This Function
        if not self.generated_embedding_ids:
            # Generate Embeddings Based On Concept Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_b ) > 0:
                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_b ) + 1, len( self.embeddings_b[1].split() ) - 1 ) )

                self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Generating Token IDs Using Embeddings" )

                # Parse Embeddings
                for index, embedding in enumerate( self.embeddings_b, 1 ):
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
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
                        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Adding Concept Token: \"" + str( embedding_text ) + "\" => Embedding Row Index: " + str( index ) )
                        self.concept_id_dictionary[embedding_text] = index
                    else:
                        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Adding Concept Token - Warning: \"" + str( embedding_text ) + "\" Already In Dictionary" )

                # Set Number Of Input Tokens Based On Token ID Dictionary Length
                self.number_of_concept_tokens  = len( self.concept_id_dictionary )

                self.embeddings_b = []
                self.embeddings_b = np.asarray( embeddings ) * scale_embedding_weight_value if scale_embedding_weight_value != 1.0 else np.asarray( embeddings )

                # Set CUI-Less Embedding To Values Close To Zero
                for idx in range( self.embeddings_b.shape[1] ): self.embeddings_b[self.concept_id_dictionary[cui_less_token]][idx] = 0.0001

                self.embeddings_b_loaded     = True
                self.generated_embedding_ids = True

        self.Print_Log( "BERTSimilarityDataLoader::Generate_Token_IDs() - Complete" )

    """
        Updates Embedding A and B In Memory Using Token And Concept ID Dictionaries
           Called When Loading Model
    """
    def Update_Token_IDs( self, data_list = [], lowercase = False ):
        # Check(s)
        if self.generated_embedding_ids == True:
            self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Error: Token & Concept Embeddings Have Aleady Been Converted/Formatted" )
            return

        self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        generated_embedding_ids = False

        # Generate Embeddings Based On Embeddings (Assumes Word2vec Format)
        # Only Generate Token ID Dictionary Using Embeddings Once.
        #   This Is Skipped During Subsequent Calls To This Function
        if not self.generated_embedding_ids:
            # Generate Embeddings Based On Term Embeddings (Assumes Word2vec Format)
            if len( self.embeddings_a ) > 0:
                # Index 0 Of Embeddings Matrix Is Padding
                embeddings = np.zeros( ( len( self.embeddings_a ) + 1, len( self.embeddings_a[1].split() ) - 1 ) )

                self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Formatted Embeddings Using Token ID Dictionary" )

                # Parse Embeddings
                for embedding in self.embeddings_a:
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
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

                self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Formatted Embeddings Using Concept ID Dictionary" )

                # Parse Embeddings
                for embedding in self.embeddings_b:
                    # Check(s)
                    if embedding == "":
                        self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Error: Embedding Contains No Data \ 'embedding == ""'", force_print = True )
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

        self.Print_Log( "BERTSimilarityDataLoader::Update_Token_IDs() - Complete" )


    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)
            file_name : File Name (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path, file_name = "" ):
        self.Print_Log( "BERTSimilarityDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTSimilarityDataLoader::Load_Vectorized_Model_Data() - Complete" )

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
        self.Print_Log( "BERTSimilarityDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTSimilarityDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Fetches NER Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        if self.lowercase_text: token = token.lower()
        self.Print_Log( "BERTSimilarityDataLoader::Get_Token_ID() - Fetching ID For Token: \"" + str( token ) + "\"" )
        return self.tokenizer.convert_tokens_to_ids( token )


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
    print( "     from NERLink.DataLoader import BERTSimilarityDataLoader\n" )
    print( "     data_loader = BERTSimilarityDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

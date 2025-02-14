#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    07/15/2020                                                                   #
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

import bioc, re, threading
import numpy as np
from sparse       import COO, concatenate, load_npz, save_npz
from scipy.sparse import csr_matrix
from bioc         import BioCAnnotation, BioCLocation

import tensorflow as tf

#tf.logging.set_verbosity( tf.logging.ERROR )                       # Tensorflow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # Tensorflow v1.x

# Tensorflow Version Check - BERT DataLoader Only Supports Tensorflow Versions >= 2.x
if re.search( r"^2.\d+", tf.__version__ ):
    import transformers
    transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
    from transformers import BertTokenizer

# Custom Modules
from NERLink.DataLoader.Base        import DataLoader
from NERLink.DataLoader.BioCreative import Passage


############################################################################################
#                                                                                          #
#   BERT Data Loader Model Class                                                           #
#                                                                                          #
############################################################################################

class BERTBioCreativeDataLoader( DataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  bert_model = "bert-base-cased", skip_individual_mentions = False, skip_composite_mentions = False, lowercase = False, ignore_label_type_list = [] ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle, lowercase = lowercase,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          skip_individual_mentions = skip_individual_mentions, skip_composite_mentions = skip_composite_mentions,
                          ignore_label_type_list = ignore_label_type_list )
        self.version                = 0.10
        self.annotation_labels      = { "O": 0, "<*>PADDING<*>" : 1 }
        self.bert_model             = bert_model
        self.max_sequence_length    = 128
        self.max_sequence_limit     = 512
        self.label_sequence_padding = 0
        self.sub_word_cls_token     = "[CLS]"
        self.sub_word_sep_token     = "[SEP]"
        self.sub_word_pad_token     = "[PAD]"
        self.sub_word_cls_token_id  = -1
        self.sub_word_sep_token_id  = -1
        self.sub_word_pad_token_id  = -1
        self.pad_label_token_id     = self.annotation_labels["<*>PADDING<*>"]
        self.pad_token_segment_id   = 0
        self.sequence_a_segment_id  = 0
        self.special_tokens_count   = 2
        self.tokenizer              = BertTokenizer.from_pretrained( self.bert_model, do_lower_case = lowercase )

        # Set [CLS] & [PAD] Token Variables Using BERT Tokenizer
        self.sub_word_cls_token_id  = self.tokenizer.convert_tokens_to_ids( self.sub_word_cls_token )
        self.sub_word_sep_token_id  = self.tokenizer.convert_tokens_to_ids( self.sub_word_sep_token )
        self.sub_word_pad_token_id  = self.tokenizer.convert_tokens_to_ids( self.sub_word_pad_token )

        # Let's Set The DataLoader's Maximum Sub-Word Sequence Length Now. This Should Be 512 For HuggingFace/PyTorch Implementation
        self.max_sequence_length    = self.tokenizer.max_model_input_sizes["bert-base-uncased"]

        # Tensorflow Version Check
        if not re.search( r"^2.\d+", tf.__version__ ):
            self.Print_Log( "BERTBioCreativeDataLoader::__init__() - Error: BERT DataLoader Only Supports Tensorflow Versions >= 2.x", force_print = True )
            exit()

    """
        Reads BIOC Formatted Data
    """
    def Read_Data( self, file_path, lowercase = True, keep_in_memory = True, encode_strings_to_utf8 = True ):
        # Check(s)
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Error: File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return self.data_list

        # Store File Path
        self.data_file_path = file_path

        # Store Lowercase Setting In DataLoader Parent Class (Used For self.Get_Token_ID() Function Calls)
        self.lowercase_text = lowercase

        data_list, annotation_type_labels = [], []

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
                        self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Warning: Passage Contains No Text Data / Skipping Passage" )
                        self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() -     ]" + str( passage ) + "'" )
                        continue

                    # Set Max Sequence Length
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
                    self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Document Passage: " + str( passage.text ) )

                    # Parse Through Passage Annotations
                    for annotation in passage.annotations:
                        if annotation.text is not None:
                            annotation_text = annotation.text.lower() if lowercase else annotation.text
                            annotation_text = self.Clean_Text( annotation_text )

                            self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Annotation Text: " + str( annotation_text ) )

                        # Parse Annotation Type
                        if "type" in annotation.infons:
                            annotation_type = annotation.infons["type"].lower()

                            # Skip Annotation Types Specified in Ignore Annotation Label Type List
                            if annotation_type in self.ignore_label_type_list:
                                self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Warning: Omitting Annotation / Annotation Type Exists In 'self.ignore_label_type_list' List" )
                                continue
                            # Parse Annotation Location If Annotation Type == 'Chemical', 'Disease', etc.
                            elif annotation_type in self.cl_accepted_labels:
                                # Check
                                if annotation_text is None:
                                    self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Warning: Annotation Contains No Text / Skipping Annotation" )
                                    continue

                                self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Type: " + str( annotation.infons["type"] ) )
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
                                    self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Location Indices: " + str( start_index ) + " to " + str( end_index ) )

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

                                    if not re.match( r'^[Mm][Ee][Ss][Hh]', annotation_concept_id ):
                                        concept_id_prefix = ","
                                    else:
                                        concept_id_prefix = ",mesh:" if lowercase else ",MESH:"

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
                                    self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Warning: Entry Term Contains No Concept Linking MeSH ID" )
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

                                self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() -   Indexing Term: " + str( entry_term ) + " => Concept ID: " + str( identifer ) )
                            # Warn User If New Annotation Type Has Been Found (Unknown Annotation Type)
                            else:
                                self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() -  Warning: New Annotation Type - " + str( annotation.infons["type"] ), force_print = True )
                        elif annotation.infons and annotation.infons.lower() not in ["type", "entry_term", "identifier", "compositemention", "individualmention"]:
                            self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() -  Warning: New Annotation Key - " + str( annotation.infons ), force_print = True )
                        else:
                            self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Warning: Annotation Contains No Type" )

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
            self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Storing Processed Data In Memory" )
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

        self.Print_Log( "BERTBioCreativeDataLoader::Read_Data() - Complete" )

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
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        if number_of_threads < 1:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Number Of Threads < 1 / Setting Number Of Threads = 1", force_print = True )
            number_of_threads = 1

        # BERT Data Loader Does Not Support CSR/COO Formats At The Moment
        if use_csr_format:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Use CSR Format Not Supported / Setting 'use_csr_format = False'" )
            use_csr_format = False

        # Enforce BERT Max Sequence Length Limitation
        if self.max_sequence_length > self.max_sequence_limit:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Max Sequence Length > " + str( self.max_sequence_limit ) + " / Enforcing BERT Max Sequence Length == "  + str( self.max_sequence_limit ) )
            self.max_sequence_length = self.max_sequence_limit

        threads             = []
        ner_inputs_ids      = []
        ner_attention_masks = []
        ner_token_type_ids  = []
        ner_label_ids       = []

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Vectorizing Data Using Settings" )
        self.Print_Log( "                                                   - Use CSR Format    : " + str( use_csr_format ) )

        total_number_of_instances = len( data_list )

        if number_of_threads > total_number_of_instances:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: 'number_of_threads > len( data_list )' / Setting 'number_of_threads = total_number_of_instances'" )
            number_of_threads = total_number_of_instances

        instances_per_thread = int( ( total_number_of_instances + number_of_threads - 1 ) / number_of_threads )

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Number Of Threads: " + str( number_of_threads ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Instances Per Thread : " + str( instances_per_thread  ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Total Instances In File Data: " + str( total_number_of_instances ) )

        ###########################################
        #          Start Worker Threads           #
        ###########################################

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Starting Worker Threads" )

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

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Waiting For Worker Threads To Finish" )

        for thread in threads:
            thread.join()

        # Convert To CSR Matrix Format
        if use_csr_format:
            ner_inputs_ids = csr_matrix( ner_inputs_ids  )
            ner_label_ids  = COO( ner_label_ids )

        if len( tmp_thread_data ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error Vectorizing Model Data / No Data Returned From Worker Threads", force_print = True )
            return None, None

        # Concatenate Vectorized Model Data Segments From Threads
        for model_data in tmp_thread_data:
            if model_data is None or len( model_data ) < 4:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error: Expected At Least Two Vectorized Lists From Worker Threads / Received None Or < 2", force_print = True )
                continue

            # Vectorized Inputs/Outputs
            encoded_input_ids, attention_mask, token_type_ids, encoded_label_ids = model_data

            # Add Encoded Input Sequence IDs To Sequence ID List
            ner_inputs_ids.extend( encoded_input_ids )
            ner_attention_masks.extend( attention_mask )
            ner_token_type_ids.extend( token_type_ids )
            ner_label_ids.extend( encoded_label_ids )

        if use_csr_format == False:
            ner_inputs_ids      = np.asarray( ner_inputs_ids )
            ner_attention_masks = np.asarray( ner_attention_masks )
            ner_token_type_ids  = np.asarray( ner_token_type_ids )
            ner_label_ids       = np.asarray( ner_label_ids )

        if isinstance( ner_inputs_ids, list ):
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Input Length  : " + str( len( ner_inputs_ids ) ) )
        elif isinstance( ner_inputs_ids, csr_matrix ) or isinstance( ner_inputs_ids, np.ndarray ):
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Input Length  : " + str( ner_inputs_ids.shape  ) )

        if isinstance( ner_label_ids, list ):
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Output Length : " + str( len( ner_label_ids ) ) )
        elif isinstance( ner_label_ids, COO ) or isinstance( ner_inputs_ids, np.ndarray ):
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Output Length : " + str( ner_label_ids.shape  ) )

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Vectorized Inputs  :\n" + str( ner_inputs_ids  ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Vectorized Outputs :\n" + str( ner_label_ids ) )

        # Clean-Up
        threads         = []
        tmp_thread_data = []

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Complete" )

        #####################
        # List Final Checks #
        #####################
        if isinstance( ner_inputs_ids, list ) and len( ner_inputs_ids ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_label_ids, list ) and len( ner_label_ids ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        ######################
        # Array Final Checks #
        ######################
        if isinstance( ner_inputs_ids, np.ndarray ) and ner_inputs_ids.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_label_ids, np.ndarray ) and ner_label_ids.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        ###########################
        # CSR Matrix Final Checks #
        ###########################
        if isinstance( ner_inputs_ids, csr_matrix ) and ner_inputs_ids.nnz == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Error: Input Matrix Is Empty" )
            return None, None

        if isinstance( ner_label_ids, COO ) and ner_label_ids.nnz == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None

        # These Can Be Saved Via DataLoader::Save_Vectorized_Model_Data() Function Call.
        if keep_in_memory:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Storing In Memory" )

            if is_validation_data:
                self.ner_val_inputs   = ( ner_inputs_ids, ner_attention_masks, ner_token_type_ids )
                self.ner_val_outputs  = ner_label_ids
            elif is_evaluation_data:
                self.ner_eval_inputs  = ( ner_inputs_ids, ner_attention_masks, ner_token_type_ids )
                self.ner_eval_outputs = ner_label_ids
            else:
                self.ner_inputs       = ( ner_inputs_ids, ner_attention_masks, ner_token_type_ids )
                self.ner_outputs      = ner_label_ids

        return ( ner_inputs_ids, ner_attention_masks, ner_token_type_ids ), ner_label_ids

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
                              mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = False,
                              use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error: No Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        prev_pad_output_setting = pad_output

        if label_per_sub_word and pad_output:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Model_Data() - Warning: Large Memory Consumption When 'label_per_sub_word = True' and 'pad_output = True' / Setting 'pad_output = False'" )
            pad_output = False

        # Clear Previous Concept Instance Data Index List
        self.concept_instance_data_idx.clear()

        # Enforce BERT Max Sequence Length Limitation
        if self.max_sequence_length > self.max_sequence_limit:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Warning: Max Sequence Length > " + str( self.max_sequence_limit ) + " / Enforcing BERT Max Sequence Length == "  + str( self.max_sequence_limit ) )
            self.max_sequence_length = self.max_sequence_limit

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Encoding Concept Instances" )

        encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks, encoded_concept_outputs = [], [], [], [], []
        output_row, output_col, output_depth, output_data = [], [], [], []
        output_row_index, output_sequence_length          = 0, 0

        for passage_idx, passage in enumerate( data_list ):
            # Encode All Term and Concept Pairs
            for annotation_tokens, annotation_idx, annotation_concepts, is_composite_mention, is_individual_mention in zip( passage.Get_Annotations(), passage.Get_Annotation_Indices(),
                                                                                                                            passage.Get_Annotation_Concept_IDs(), passage.Get_Composite_Mention_List(),
                                                                                                                            passage.Get_Individual_Mention_List() ):
                if annotation_tokens == "" or annotation_concepts == "":
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Warning: Instance Contains No Entry Terms Or Concept IDs" )
                    self.Print_Log( "                                                  -          Sequence: " + str( passage.Get_Passage() ) )
                    continue
                elif is_composite_mention and self.skip_composite_mentions:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Composite Mention Detected / Skipping Composite Mention" )
                    continue
                elif is_individual_mention and self.skip_individual_mentions:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Individual Mention Detected / Skipping Individual Mention" )
                    continue

                # Tokenize The Sequence Using The BERT Tokenizer And Extract The Tokenized Sequence Information (IDs Per Concept Entry-Term Sub-Word)
                encoded_input_instance, encoded_output_instance = self.Encode_CL_Instance( text_sequence = passage.Get_Passage(), entry_term = annotation_tokens,
                                                                                           annotation_concept = annotation_concepts, annotation_indices = annotation_idx,
                                                                                           pad_input = pad_input, pad_output = pad_output, term_sequence_only = term_sequence_only,
                                                                                           concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                                                           separate_sentences = separate_sentences, restrict_context = restrict_context,
                                                                                           label_per_sub_word = label_per_sub_word )

                if encoded_input_instance is None or encoded_output_instance is None:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Input And/Or Output Instance" )
                    continue

                if label_per_sub_word and output_sequence_length == 0: output_sequence_length = len( encoded_output_instance )

                # Extract Encoded Input Elements From 'encoded_input_instance' Tuple
                token_ids, attention_mask, token_type_ids, entry_term_mask = encoded_input_instance

                # Store Encoded Instance Elements In Their Appropriate Lists
                encoded_token_ids.append( token_ids )
                encoded_attention_masks.append( attention_mask )
                encoded_token_type_ids.append( token_type_ids )
                encoded_entry_term_masks.append( entry_term_mask )

                # Concept Output Is A Vector/Array Of 'N' Classes With Our Desired Instance Class As '1'
                #   i.e. [0, 0, 1, 0, 0]
                if use_csr_format:
                    for i, value in enumerate( encoded_output_instance ):
                        if not label_per_sub_word and value == 0: continue

                        # Pad CL Output
                        if label_per_sub_word and prev_pad_output_setting:
                            # Value Is List Of Concept IDs - Complex Mention
                            if isinstance( value, list ):
                                for val in value:
                                    output_row.append( output_row_index )
                                    output_col.append( i )
                                    output_depth.append( val )
                                    output_data.append( 1 )
                            # Value Is Single Concept ID - Singular Mention
                            else:
                                output_row.append( output_row_index )
                                output_col.append( i )
                                output_depth.append( value )
                                output_data.append( 1 )
                        # Non-Padded CL Output
                        else:
                            output_row.append( output_row_index )
                            output_col.append( i )
                            output_data.append( value )

                    output_row_index += 1
                else:
                    encoded_concept_outputs.append( encoded_output_instance )

                # Keep Track Of Which Passage The Instance Came From
                self.concept_instance_data_idx.append( passage_idx )

        # Convert Data To Numpy Arrays
        encoded_token_ids        = np.asarray( encoded_token_ids,        dtype = np.int32 )
        encoded_attention_masks  = np.asarray( encoded_attention_masks,  dtype = np.int32 )
        encoded_token_type_ids   = np.asarray( encoded_token_type_ids,   dtype = np.int32 )
        encoded_entry_term_masks = np.asarray( encoded_entry_term_masks, dtype = np.int32 )

        # Convert Into CSR_Matrix
        if use_csr_format:
            output_data      = np.asarray( output_data, dtype = np.int32 )
            number_of_labels = self.Get_Number_Of_Unique_Concepts() if prev_pad_output_setting else 1

            if label_per_sub_word and prev_pad_output_setting:
                encoded_concept_outputs = COO( [ output_row, output_col, output_depth ], output_data, shape = ( output_row_index, output_sequence_length, number_of_labels ), fill_value = 0 )
            else:
                encoded_concept_outputs = COO( [ output_row, output_col ], output_data, shape = ( output_row_index, number_of_labels ), fill_value = 0 )
        else:
            encoded_concept_outputs = np.asarray( encoded_concept_outputs, dtype = np.int32 )

        # Check(s)
        number_of_input_instances  = encoded_token_ids.shape[0]
        number_of_output_instances = encoded_concept_outputs.shape[0] if isinstance( encoded_concept_outputs, COO ) else len( encoded_concept_outputs )

        if number_of_input_instances == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Inputs", force_print = True )
            return None, None
        elif number_of_output_instances == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error Occurred While Encoding Concept Outputs", force_print = True )
            return None, None
        elif number_of_input_instances != number_of_output_instances:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error: Number Of Input And Output Instances Not Equal", force_print = True )
            return None, None

        if keep_in_memory:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Storing Encoded Data In Memory" )

            if is_validation_data:
                self.concept_val_inputs   = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_val_outputs  = encoded_concept_outputs
            elif is_evaluation_data:
                self.concept_eval_inputs  = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_eval_outputs = encoded_concept_outputs
            else:
                self.concept_inputs       = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks )
                self.concept_outputs      = encoded_concept_outputs

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Complete" )

        return ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_entry_term_masks ), encoded_concept_outputs

    """
        Tokenizes Data Sequences Into List Of Tokens With Or Without Padding
            Used For ELMo Implementation

        Input:
            data_list           : List Of Variables
            use_padding         : Pads Each Sequence To 'self.max_sequence_length' Token Elements

        Output:
            tokenized_sequences : List Of Tokenized Sequences
    """
    def Tokenize_Model_Data( self, data_list = [], use_padding = True ):
        # If The User Does Not Pass Any Data, Try The Data Stored In Memory (DataLoader Object)
        if len( data_list ) == 0 and len( self.data_list ) > 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Tokenize_Model_Data() - Warning: No Data Specified, Using Data Stored In Memory" )
            data_list = self.data_list

        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Tokenize_Model_Data() - Error: Data List Is Empty", force_print = True )
            return []

        self.Print_Log( "BERTBioCreativeDataLoader::Tokenize_Model_Data() - Tokenizing Data Sequences" )
        self.Print_Log( "BERTBioCreativeDataLoader::Tokenize_Model_Data() -   Use Padding: " + str( use_padding ) )

        tokenized_sequences = []

        # Split Each Passage Text Sequence By White Space And Store As List Of Tokens
        for passage in data_list:
            passage_length = len( passage.Get_Passage().split() )
            sequence = [ "" for _ in range( self.Get_Max_Sequence_Length() ) ] if use_padding else [ "" for _ in range( passage_length ) ]

            for index, token in enumerate( passage.Get_Passage().split() ):
                sequence[index] = token

            # Store Tokenized Sequence
            tokenized_sequences.append( sequence )

        self.Print_Log( "BERTBioCreativeDataLoader::Tokenize_Model_Data() - Complete" )

        return tokenized_sequences

    """
        Returns List Of Strings, Compiled From Data Sequences
            Used For ELMo Implementation

        Input:
            data_list : List Of Variables

        Output:
            sequences : List Of Sequences
    """
    def Get_Data_Sequences( self, data_list = [] ):
        # If The User Does Not Pass Any Data, Try The Data Stored In Memory (DataLoader Object)
        if len( data_list ) == 0 and len( self.data_list ) > 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Get_Data_Sequences() - Warning: No Data Specified, Using Data Stored In Memory" )
            data_list = self.data_list

        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Get_Data_Sequences() - Error: Data List Is Empty", force_print = True )
            return []

        self.Print_Log( "BERTBioCreativeDataLoader::Get_Data_Sequences() - Fetching Data Sequences" )

        # Split Each Passage Text Sequence By White Space And Store As List Of Tokens
        sequences = [ passage.Get_Passage() for passage in data_list ]

        self.Print_Log( "BERTBioCreativeDataLoader::Get_Data_Sequences() - Complete" )

        return sequences

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            text_sequence            : Sequence Of Text (String)
            annotations              : List Of Term Annotations (List Of Strings)
            annotation_labels        : List Of Term Labels (List Of Strings)
            annotation_indices       : List Of Annotation Indices (List Of xx:xx Strings)
            composite_mention_list   : List Of Boolean Identifying If Annotations Are Composite Mentions (List Of Bools)
            individual_mention_list  : List Of Boolean Identifying If Annotations Are Individual Mentions (List Of Bools)
            use_padding              : Adds Padding To Input Sequence ie. BERT [PAD] Tokens After Actual Sequence Until Max Sequence Length (Bool)
            limit_to_max_sequence    : True -> Limit/Truncate Sequence To BERT Tokenizer's Sub-Word Limit, False -> No Limit
            skip_individual_mention  : Skips Individual Annotation Mentions (Bool)
            skip_composite_mention   : Skips Composite Annotation Mentions (Bool)

        Outputs:
            encoded_text_id_sequence : Encoded Text Sequence List i.e. List Of Sequence IDs
            attention_mask           : Attention Mask List
            token_type_ids           : Encoded Token ID List
            encoded_label_sequence   : Encoded Sequence Token Entity Label List (Sparse Categorical Crossentropy Format)
    """
    def Encode_NER_Instance( self, text_sequence, annotations, annotation_labels, annotation_indices, composite_mention_list = [], individual_mention_list = [],
                             use_padding = True, limit_to_max_sequence = True, skip_composite_mentions = True, skip_individual_mentions = False ):
        # Check(s)
        if text_sequence == "" or len( text_sequence ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() - Error: Text Sequence Is Empty String" )
            return [], [], [], []

        if len( annotations ) == 0 or len( annotation_labels ) == 0:
            self.Print_Log( "BERTBioCteativeDataLoader::Encode_NER_Instance() - Error: 'annotations' or 'annotation_labels' == 0" )
            return [], [], [], []

        encoded_label_sequence = [ 0 for _ in range( len( text_sequence.split() ) ) ]

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() - Encoding Inputs" )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() -                   Text Sequence         : " + str( text_sequence         ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() -                   Annotations           : " + str( annotations           ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() -                   Annotation Indices    : " + str( annotation_indices    ) )

        # Determine Which Entities Have Annotations
        curr_annotation_index        = 0
        curr_offset                  = 0
        annotation_id_to_idx         = {}
        text_sequence_character_mask = [ "_" if text_sequence[i] != " " else " " for i in range( len( text_sequence ) ) ]

        for idx, ( annotation, annotation_idx, is_composite_mention, is_individual_mention ) in enumerate( zip( annotations, annotation_indices,
                                                                                                                composite_mention_list, individual_mention_list ) ):
                # Skip Individual Mentions / Skip Composite Mentions
                if self.Get_Skip_Individual_Mentions() and is_individual_mention or self.Get_Skip_Composite_Mentions() and is_composite_mention: continue

                number_of_indices = len( annotation_idx.split( "<:>" ) )

                for indices in annotation_idx.split( "<:>" ):
                    indices           = indices.split( ":" )
                    annotation_offset = int( indices[0]  )
                    annotation_end    = int( indices[-1] )

                    # Extract Annotation Token Using Token Indices (Debugging/Testing)
                    extracted_token   = text_sequence[annotation_offset:annotation_end]

                    if number_of_indices == 1 and annotation != extracted_token:
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error: Extracted Token != True Token", force_print = True )
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() -        True Token: " + str( annotation ), force_print = True )
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() -        Extracted Token: " + str( extracted_token ), force_print = True )
                        continue
                    elif number_of_indices > 1 and extracted_token not in annotation:
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() - Error: Extracted Token Not In True Token", force_print = True )
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() -        True Token: " + str( annotation ), force_print = True )
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Model_Data() -        Extracted Token: " + str( extracted_token ), force_print = True )
                        continue

                    # Add Current Offset To Extracted Offset For Character Mask
                    annotation_offset += curr_offset
                    annotation_end    += curr_offset

                    index_size = len( str( curr_annotation_index ) )

                    # Expand The Character Mask By The Size Of The Current Annotation Index If Index Size Exceeds Token Length
                    #   NOTE: Assumes First Character Is "_". If Not, Then This Will Produce Errors.
                    if text_sequence_character_mask[annotation_offset:annotation_offset + index_size][-1] == " ":
                        temp_char_list             = text_sequence_character_mask[annotation_offset:annotation_offset + index_size]
                        underscore_character_count = 0

                        for character in temp_char_list:
                            if character == "_": underscore_character_count += 1

                        increase_by_length = len( temp_char_list ) - underscore_character_count
                        text_sequence_character_mask.insert( annotation_offset, "_" * increase_by_length )
                        curr_offset += increase_by_length

                    text_sequence_character_mask[annotation_offset:annotation_offset + index_size] = str( curr_annotation_index )
                    text_sequence_character_mask[annotation_offset + index_size:annotation_end]    = ["#"] * ( annotation_end - annotation_offset - index_size )

                    # Double Check To See If The Annotation Doesn't Contain Spaces (Annotation Not Composed Of Multiple Singular Terms)
                    space_indices = [i for i, char in enumerate( extracted_token ) if char == " "]

                    # Fill Spaces Back In
                    for index in space_indices:
                        offset_index = annotation_offset + index
                        text_sequence_character_mask[offset_index]     = " "
                        text_sequence_character_mask[offset_index + 1] = str( curr_annotation_index )

                annotation_id_to_idx[curr_annotation_index] = idx
                curr_annotation_index += 1

        text_sequence_character_mask     = '' . join( text_sequence_character_mask )
        text_sequence_annotation_indices = text_sequence_character_mask.split()

        ###############################################
        # Encode Text Sequence And Set Attention Mask #
        ###############################################

        prev_annotation_id = 0
        subword_tokens, processed_annotation_ids, subword_label_ids = [], [], []

        # Tokenize Data Into Sub-Words And Assign Label IDs Per Sub-Word
        for ( index, token ), token_annotation_id in zip( enumerate( text_sequence.split() ), text_sequence_annotation_indices ):
            token_subwords = self.tokenizer.tokenize( token )
            subword_tokens.extend( token_subwords )

            annotation_ids           = []
            temp_token_annotation_id = token_annotation_id
            curr_token_annotation_id = -1

            # Identify Possible Annotations In 'temp_token_annotation_id'
            while True:
                # Keep Finding & Storing Annotation IDs Until None Exist
                size_of_prev_annotation_id = len( str( prev_annotation_id ) )
                match = re.search( r'(\d+)', temp_token_annotation_id )
                if match:
                    curr_token_annotation_id      = match.group( 1 )
                    curr_token_annotation_id_list = []

                    # Break-Up IDs Right Next To Each Other i.e. '2021' To '20', Then '21'.
                    #   If The Previous Digit Size Is Less Than Two Sizes Of The Matched Text
                    if len( curr_token_annotation_id ) > size_of_prev_annotation_id + 1:
                        curr_token_annotation_id = curr_token_annotation_id[0:size_of_prev_annotation_id]

                    # Add Current Token Annotation ID Where The Matched Values Of The Last Digit Are Between '0' to '9'
                    if len( curr_token_annotation_id ) == size_of_prev_annotation_id and str( prev_annotation_id )[-1] < '9':
                        curr_token_annotation_id_list.append( curr_token_annotation_id )
                    # Account For Incrementing The Number Of Annotation Token Digits By One
                    elif len( curr_token_annotation_id ) == size_of_prev_annotation_id + 1 and str( prev_annotation_id )[-1] == '9':
                        curr_token_annotation_id_list.append( curr_token_annotation_id )
                    # Separate Numbers In Matched Text By Size Of Previous Annotation ID
                    else:
                        curr_token_annotation_id = curr_token_annotation_id[0:size_of_prev_annotation_id]
                        curr_token_annotation_id_list.append( curr_token_annotation_id )

                    for val in curr_token_annotation_id_list:
                        if prev_annotation_id < int( val ): prev_annotation_id = int( val )
                        temp_token_annotation_id = re.sub( r'' + str( val ) + '', "", temp_token_annotation_id )
                        annotation_ids.append( val )
                else:
                    break

            # Assign Label ID Per Sub-Word Of Given Token
            if len( annotation_ids ) > 0:
                # Fill Annotation Labels With 'O' Token ID Of B-I-O NER Label Format
                subword_label_ids.extend( [self.annotation_labels["O"]] * len( token_subwords ) )

                # For Each Extracted Annotation In The Current Token (Or List Of Token Sub-Words)
                #   Replace The Appropriate Index Of 'subword_label_ids' With It's Desired Label
                for annotation_id in annotation_ids:
                    # Get Length Of Term/Annotation Text
                    match = None

                    if str( annotation_id ) + "#" in token_annotation_id:
                        # Annotation ID Consisting Of Number Followed By '#' Characters i.e. '0####'
                        match = re.search( r'(' + str( annotation_id ) + '#+)', token_annotation_id )
                    else:
                        # Annotation ID Only Consisting Of The Number/ID i.e. '0'
                        match = re.search( r'(' + str( annotation_id ) + ')', token_annotation_id )

                    # Determine Start And End Indices Which Designates Where The Annotation Occurs Within The Term
                    annotation_length = len( match.group( 1 ) )
                    annotation_start  = token_annotation_id.index( annotation_id )
                    annotation_id     = int( annotation_id )

                    # Get Annotation Label
                    subword_label = annotation_labels[annotation_id_to_idx[annotation_id]]

                    temp_word, start_tagged = "", False if annotation_id not in processed_annotation_ids else True
                    current_idx = len( subword_label_ids ) - ( len( token_subwords ) )

                    # Tag The Specific Start & End Indices With The Appropriate NER Label
                    for token_idx, sub_word in enumerate( token_subwords ):
                        if sub_word != "#": sub_word = re.sub( r'^#+', "", sub_word )
                        temp_word        += sub_word
                        temp_word_length = len( temp_word ) - 1 - annotation_start

                        # If Annotation Start Index Equals Or Within Current Sub-Word, Tag Sub-Word Token As Beginning Label
                        if not start_tagged and len( temp_word ) - 1 >= annotation_start:
                            subword_label_ids[current_idx + token_idx] = int( self.annotation_labels["B-" + str( subword_label )] )
                            start_tagged = True
                        # Continue Tagging Next Sub-Word Tokens As 'Intermediate' Label Of Preceeding Label
                        elif start_tagged and len( temp_word ) - 1 >= annotation_start and temp_word_length < annotation_length:
                            subword_label_ids[current_idx + token_idx] = int( self.annotation_labels["I-" + str( subword_label )] )
                        # We've Exceeded The Actual Annotation Label Span, Discontinue Modifying NER Label As 'Beginning' Or 'Intermediate'
                        elif temp_word_length > annotation_length:
                            break

                    # Add Current Annotation ID To Processed Annotation IDs
                    if annotation_id not in processed_annotation_ids: processed_annotation_ids.append( annotation_id )
            else:
                subword_label_ids.extend( [self.annotation_labels["O"]] * len( token_subwords ) )

        # Make Room To Add The [CLS] and [SEP] Special Tokens
        if limit_to_max_sequence and len( subword_tokens ) > self.max_sequence_length - self.special_tokens_count:
            subword_tokens    = subword_tokens[: ( self.max_sequence_length - self.special_tokens_count ) ]
            subword_label_ids = subword_label_ids[: ( self.max_sequence_length - self.special_tokens_count ) ]

        # Encode Sub-Word Tokens Into BERT Inputs: Token IDs, Attention Masks and Token Type IDs
        max_length = self.max_sequence_length if limit_to_max_sequence else len( subword_tokens ) + self.special_tokens_count
        inputs     = self.tokenizer.encode_plus( subword_tokens, add_special_tokens = True, max_length = max_length )

        encoded_text_id_sequence, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        encoded_label_sequence = [self.label_sequence_padding] + subword_label_ids + [self.label_sequence_padding]

        # Pad Sequences
        if use_padding and self.max_sequence_length - len( encoded_text_id_sequence ) > 0:
            padding_length      = self.max_sequence_length - len( encoded_text_id_sequence )
            padding_token_value = self.annotation_labels["O"]
            encoded_text_id_sequence.extend( [self.sub_word_pad_token_id] * padding_length )
            attention_mask.extend( [self.sub_word_pad_token_id] * padding_length )
            token_type_ids.extend( [self.sub_word_pad_token_id] * padding_length )
            encoded_label_sequence.extend( [padding_token_value] * padding_length )

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_NER_Instance() - Complete" )

        return encoded_text_id_sequence, attention_mask, token_type_ids, encoded_label_sequence

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            text_sequence         : Text Sequence In Which The 'entry_term' Occurs. (String)
            entry_term            : Concept Token (String)
            annotation_indices    : Concept Token Indices (String Of Two Integers Separated By ':' Character)
            pad_input             : Adds Padding To Input Sequence ie. [PAD] Tokens After Actual Sequence Until Max Sequence Length (Bool)
            mask_term_sequence    : True  -> Entry Term Mask = Entire Sub-Word Sequence Containing Entry Term
                                    False -> Encode Just Entry Term Sub-Word Tokens
            separate_sentences    : Separates Sentences With [SEP] Token Using Sentence Delimiters (Bool)
            term_sequence_only    : Disregards All Sequences Surrounding The Sequence Of Interest, Only Encoding The Desired Sequence (Bool)
            restrict_context      : Restricts Or Reduces The Sequence Context Surrounding The Entry Term Used To Generate Its Embedding (Bool)
            limit_to_max_sequence : True -> Limit/Truncate Sequence To BERT Tokenizer's Sub-Word Limit, False -> No Limit

        Outputs:
            encoded_entry_term    : Tuple Consisting Of:
                                          1) Encoded Text Sequence Sub-Word IDs
                                          2) Attention Masks
                                          3) Token Type IDs
                                          4) Entry Term Mask
    """
    def Encode_CL_Input_Instance( self, text_sequence, entry_term, annotation_indices, pad_input = True, mask_term_sequence = False,
                                  separate_sentences = True, term_sequence_only = False, restrict_context = False, limit_to_max_sequence = True ):
        # Check(s)
        # 'mask_term_sequence = True' Depends On 'separate_sentences = True'
        if mask_term_sequence and not separate_sentences:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Warning: 'mask_term_sequence = True' Depends on 'separate_sentences = True'" )
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -           Setting 'separate_sentences = True" )
            separate_sentences = True

        # 'term_sequence_only = True' Depends On 'restrict_context = True'
        if term_sequence_only and not restrict_context:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Warning: 'term_sequence_only = True' Depends on 'restrict_context = True'" )
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -           Setting 'restrict_context = True" )
            restrict_context = True

        # 'term_sequence_only = True' Depends On 'separate_sentences = True'
        if term_sequence_only and not separate_sentences:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Warning: 'term_sequence_only = True' Depends on 'separate_sentences = True'" )
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -           Setting 'separate_sentences = True" )
            separate_sentences = True

        # Placeholders For Vectorized Inputs/Outputs (Pad Remaining Tokens Outside Of Sequence Length)
        encoded_text_id_sequence, attention_mask, token_type_ids, entry_term_mask = [], [], [], []

        # Map Entry Terms To Concepts
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Encoding Text Sequence/Concept To Concept ID & Masking Entry Term Sub-Words" )

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Text Sequence: " + str( text_sequence      ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Entry Term   : " + str( entry_term         ) )

        # -------------------------------------------------------------- #
        # Determine Where The Entry Term Exists Within The Text Sequence #
        # -------------------------------------------------------------- #
        term_indices   = annotation_indices.split( ":" )
        extracted_term = text_sequence[ int( term_indices[0] ):int( term_indices[-1] ) ]

        if extracted_term != entry_term:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: Extracted Term != Entry Term" )
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -        Entry Term    : " + str( entry_term     ) )
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -        Extracted Term: " + str( extracted_term ) )
            return None

        # Determine How Many Splits Occur Before The Extracted Token
        preceeding_sequence         = text_sequence[ 0:int( term_indices[0] ) ]
        number_of_preceeding_splits = len( preceeding_sequence.split() )

        # Adjust For Last Element Not Being A Space i.e. 'entry_term == mdma' and 'token == (mdma,'
        if len( preceeding_sequence ) > 0 and preceeding_sequence[-1] != " ": number_of_preceeding_splits -= 1

        # Determine If The Entry Term Is A Multi-Word Term
        is_multi_word_term = True if len( entry_term.split() ) > 0 else False

        # ------------------------------------------- #
        # Encode Text Sequence And Set Attention Mask #
        # ------------------------------------------- #

        # Separate Sentences With [SEP] Token
        if separate_sentences:
            text_sequence       = self.Separate_Sequences_With_SEP_Token( text_sequence )
            preceeding_sequence = self.Separate_Sequences_With_SEP_Token( preceeding_sequence )

            # Re-Count Preceeding Splits Before Desired Entry Term
            number_of_preceeding_splits = len( preceeding_sequence.split() )

            # Adjust For Last Element Not Being A Space i.e. 'entry_term == mdma' and 'token == (mdma,'
            if len( preceeding_sequence ) > 0 and preceeding_sequence[-1] != " ": number_of_preceeding_splits -= 1

        subword_tokens, curr_entry_term, sub_word_found = [], "", False,
        entry_term_subword_length                       = 0

        # Tokenize Data Into Sub-Words And Assign Label IDs Per Sub-Word
        for index, token in enumerate( text_sequence.split() ):
            # Tokenize Single Token In To Sub-Word Tokens
            token_subwords = self.tokenizer.tokenize( token )

            # Add Tokenized Sub-Words To Our Complete Sequence Sub-Word List
            subword_tokens.extend( token_subwords )

            # Create A Mask List Of '1' Values Where Our Entry Term Exists Within The Text Sequence, '0' Otherwise
            token_mask = []

            # Token Is The Desired Entry Term. Let's Build The Appropriate Term Mask Of The
            #   Exact Entry Term Without Without Any Other Surrounding Characters.
            if index == number_of_preceeding_splits or ( is_multi_word_term and sub_word_found ):
                for sub_word_idx, sub_word in enumerate( token_subwords ):
                    # Remove '#' Character From Sub-Words After First Sub-Word In A Given Token
                    cleaned_sub_word      = re.sub( '^#+', "", sub_word ) if sub_word != "#" else sub_word
                    next_cleaned_sub_word = re.sub( '^#+', "", token_subwords[sub_word_idx + 1] ) if sub_word_idx < len( token_subwords ) - 1 else None    # Not Currently Used

                    # Determine Current Number Of Splits Within The Current Entry Term
                    #   This Is The Current Multi-Token Index Given Our Entry Term Is Multi-Token
                    whitespace_split_count = len( curr_entry_term.split() ) - 1 if len( curr_entry_term.split() ) > 1 else 0

                    # Find The Beginning Of Our Desired Entry Term Using It's Sub-Words
                    if curr_entry_term != entry_term and not sub_word_found:
                        # Since We Cannot Guarantee The Cleaned Sub-Word Does Not Contain Characters Which Are Not Specified Within Our
                        #   Entry Term Span We Must Check All Characters Within The Cleaned Sub-Word Against The Entry Term.
                        for char_idx, sub_word_character in enumerate( cleaned_sub_word ):
                            next_sub_word_character = cleaned_sub_word[char_idx + 1] if char_idx < len( cleaned_sub_word ) - 1 else None

                            # Find Begining Of Entry Term In Sub-Word
                            if next_sub_word_character is None and re.search( r'^' + re.escape( sub_word_character ), entry_term ) or \
                               next_sub_word_character is not None and re.search( r'^' + re.escape( sub_word_character + next_sub_word_character ), entry_term ):
                                curr_entry_term += sub_word_character
                                sub_word_found   = True
                            # Add Remaining Characters If Sub-Word Found
                            elif len( curr_entry_term ) > 0 and \
                                 ( next_sub_word_character is None and re.search( re.escape( curr_entry_term + sub_word_character ), entry_term ) or \
                                   next_sub_word_character is not None and re.search( re.escape( curr_entry_term + sub_word_character + next_sub_word_character ), entry_term ) ):
                                curr_entry_term += sub_word_character

                            # Double Check For Current Sub Word == Entry Term Condition
                            #   i.e. If Entry Term is A Sub-set Of The Current Sub-Word Or Token
                            if curr_entry_term == entry_term: break

                        # We Found The Entire Entry Term
                        #   Mark It in The Entry Term List
                        if curr_entry_term == entry_term:
                            token_mask.extend( [1] )
                            sub_word_found = False
                            entry_term_subword_length += 1
                        # We've Found The Beginning Of Our Desired Entry Term
                        #   Mark It in The Entry Term List
                        elif sub_word_found:
                            token_mask.extend( [1] )
                            entry_term_subword_length += 1
                        # We Didn't Find The Beginning Of Our Desired Entry Term
                        else:
                            sub_word_found = False
                            token_mask.extend( [0] )
                    # Continue Identifying Our Entry Term, Building Its Mask (Values == '1') Until Our Entire Term Is Found
                    elif sub_word_found and curr_entry_term + cleaned_sub_word[0] in entry_term or \
                         sub_word_found and re.search( re.escape( curr_entry_term ) + r'\s+' + re.escape( cleaned_sub_word[0] ), entry_term ):
                        # Check/Account For Multi-Token Term (Whitespace In-Between Tokens)
                        if re.search( r'\s+', entry_term ) and curr_entry_term.split() == entry_term.split()[0:whitespace_split_count+1]:
                            whitespace       = re.findall( r'\s+', entry_term )[whitespace_split_count]
                            curr_entry_term += whitespace

                        # Add One Character At A Time Until We Find The End Of The Entry Term
                        for char_idx, sub_word_character in enumerate( cleaned_sub_word ):
                            next_sub_word_character = cleaned_sub_word[char_idx + 1] if char_idx < len( cleaned_sub_word ) - 1 else None

                            # Keep Adding Characters By Checking If The Additional Character With Our Current
                            #   Entry Term Exists Within The True Entry Term
                            if next_sub_word_character is None and re.search( r'^' + re.escape( curr_entry_term + sub_word_character ), entry_term ) or \
                               next_sub_word_character is not None and re.search( r'^' + re.escape( curr_entry_term + sub_word_character + next_sub_word_character ), entry_term ):
                                curr_entry_term += sub_word_character
                            # The Character Does Not Exist Within Our Entry Term. Remove The False Positive Characters
                            #   And Continue Looking In The Next Sub-Word Tokens.
                            else:
                                curr_entry_term = ""
                                sub_word_found  = False

                            # Double Check For Current Sub Word == Entry Term Condition
                            #   i.e. If Entry Term is A Sub-set Of The Current Sub-Word Or Token
                            if curr_entry_term == entry_term: break

                        # Sub-Word Exists In Entry Term
                        if sub_word_found:
                            token_mask.extend( [1] )
                            entry_term_subword_length += 1
                        # Sub-Word Does Not Exist Within Entry Term
                        else:
                            token_mask.extend( [0] )
                    # We Incorrectly Selected The Wrong Beginning Sub-Word
                    elif sub_word_found and curr_entry_term + cleaned_sub_word[0] not in entry_term:
                        # If The Sub-Word Index Is > 0, Then Means We've Been Incorrectly Selecting Previous Sub-Words
                        #   Zero Them Out In Our Entry Term Mask
                        if sub_word_idx > 0:
                            token_mask      = [ 0 for val in token_mask ]
                            entry_term_mask = [ 0 for val in entry_term_mask]
                        sub_word_found                 = False
                        curr_entry_term                = ""
                        entry_term_subword_length      = 0
                        token_mask.extend( [0] )
                    # If The Sub-Word Is Not A Part Of Our Entry Term, Then It's Entry Term Mask Value Is '0'
                    else:
                        token_mask.extend( [0] )

                    # If We've Found The Complete Sub-Word Set The Found Flag To False.
                    #   This Will Set All Remaining Sub-Words To '0' In Our Entry Term Mask.
                    if curr_entry_term == entry_term:
                        sub_word_found = False
            # Token Is Not A Desired Entry Term, Fill Associated Sub-Word Indices With '0's.
            else:
                token_mask = [0] * len( token_subwords )

            entry_term_mask.extend( token_mask )

        # Check To See If The Entry Term Mask Contains Our Desired Entry Term.
        #   If It's All '0's, Then It Doesn't. We Should Skip This Instance Since It'll Zero-Out
        #   The Network By Passing An Embedding Of All Zero Elements.
        if 1 not in entry_term_mask:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: Entry Term Mask Is All Zeros" )
            self.Print_Log( "                                                - Sequence           : " + str( text_sequence ) )
            self.Print_Log( "                                                - Preceding Sequence : " + str( preceeding_sequence ) )
            self.Print_Log( "                                                - # Preceding Splits : " + str( number_of_preceeding_splits ) )
            self.Print_Log( "                                                - Entry Term         : " + str( entry_term ) )
            self.Print_Log( "                                                - Captured Entry Term: " + str( curr_entry_term ) )
            return None

        # Determine Entry Term Mask Index Boundaries
        #   'token_mask_sub_word_length' Also Accounts For Entry Term Sub-Words Ending At The End Of The Sequence
        #    i.e. 'token_mask_sub_word_length = len( entry_term_mask ) - 1 - token_mask_sub_word_start_idx
        token_mask_sub_word_start_idx = entry_term_mask.index( 1 )
        token_mask_sub_word_length    = entry_term_mask.index( 0, token_mask_sub_word_start_idx ) - token_mask_sub_word_start_idx if entry_term_mask[-1] == 0 else len( entry_term_mask ) - 1 - token_mask_sub_word_start_idx

        # Now For The Tricky Part: Determine If The Entry Term Mask Exists Within The Max Sequence Length.
        #                          If So, We're Fine. Concatenate The Remaining Tokens And Proceed With
        #                          Encoding The Input Data.
        #                          If Not, We Must Shift Our Tokens To Surround The Entry Term To Contextualize
        #                          The Sub-Words Surrounding The Entry Term.
        if limit_to_max_sequence and len( subword_tokens ) >= self.max_sequence_length - self.special_tokens_count:  # Make Room To Add The [CLS] and [SEP] Special Tokens
            sub_word_buffer_size = 50

            # If Entry Term Occurs < 'self.max_sequence_length - sub_word_buffer_size', Proceed With Encoding Data.
            if token_mask_sub_word_start_idx + token_mask_sub_word_length < self.max_sequence_length - sub_word_buffer_size:
                subword_tokens  = subword_tokens[: ( self.max_sequence_length - self.special_tokens_count ) ]
                entry_term_mask = entry_term_mask[: ( self.max_sequence_length - self.special_tokens_count ) ]
            # Shift Our Tokenized Input Sequence To Surround The Entry-Term Using Its Index. (Sentence-Based)
            else:
                start_index = token_mask_sub_word_start_idx + int( token_mask_sub_word_length / 2 ) - 250 if token_mask_sub_word_start_idx + int( token_mask_sub_word_length / 2 ) - 200 > 1 else 1
                end_index   = token_mask_sub_word_start_idx + token_mask_sub_word_length + 350 if token_mask_sub_word_start_idx + token_mask_sub_word_length + 350 < len( subword_tokens ) - 1 else len( subword_tokens ) - 1
                if end_index - start_index > 510: end_index = start_index + 510
                subword_tokens  = subword_tokens[ start_index : end_index ]
                entry_term_mask = entry_term_mask[ start_index : end_index ]

                # Find Starting Index Of New Token Mask Sub-Word
                token_mask_sub_word_start_idx = entry_term_mask.index( 1 )

            # Double Check To See If The Entry Term Mask Contains Our Desired Entry Term.
            #   If It's All '0's, Then It Doesn't. We Should Skip This Instance Since It'll Zero-Out
            #   The Network By Passing An Embedding Of All Zero Elements.
            if 1 not in entry_term_mask:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: Shifting Array Surrounding Entry Term Removed Term From Encoded Sequence / Entry Term Mask Is All Zeros" )
                return None

        # Restrict Entry Term Context
        #   Locates Previous [SEP] Token To Use Sequences Before And After Our Sequence Containing The 'entry_term'
        #   As The Context To Generate Our Desired 'average', 'first' or 'last' Entry-Term Sub-Word Embeddings.
        num_subwords_before_entry_term = token_mask_sub_word_start_idx if token_mask_sub_word_start_idx >= 0 else -1

        if restrict_context and num_subwords_before_entry_term != -1:
            if separate_sentences:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Restricting Term Context Using [SEP] Tokens" )
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -   Sequence Sub-Words: " + str( subword_tokens ) )

                # Locate Beginning Of Previous Sentence/Sequence
                start_index, end_index = 0, 0
                sep_token_offset       = 1 if term_sequence_only else 2
                sep_buffer_size        = 1 if term_sequence_only else 2

                # Find [SEP] Tokens In The Sub-Word Elements Prior To The Entry Term Index.
                #   When Found, Use The Second To Last [SEP] + 1 Sub-Word Index As Our Desired Starting Index To
                #   Constrain Our Sub-Word Sequence
                if token_mask_sub_word_start_idx > 0:
                    sep_indices = [ idx for idx, token in enumerate( subword_tokens[ 0 : num_subwords_before_entry_term ] ) if token == "[SEP]" ]
                    start_index = 0 if len( sep_indices ) < sep_buffer_size else sep_indices[-sep_token_offset] + 1
                else:
                    start_index = 0

                # Find Next Sequence Index End (i.e. [SEP] Token) After Our Current Sequence Of Interest.
                #   We Use This As The End Index To Determine The Bounds Of Our Entry Term Context.
                if num_subwords_before_entry_term + 1 < len( subword_tokens ):
                    sep_indices = [ idx for idx, token in enumerate( subword_tokens[ num_subwords_before_entry_term : len( subword_tokens ) ] ) if token == "[SEP]" ]
                    end_index   = len( subword_tokens ) if len( sep_indices ) < sep_buffer_size else sep_indices[sep_token_offset-1] + num_subwords_before_entry_term
                else:
                    end_index   = len( subword_tokens )

                subword_tokens  = subword_tokens[ start_index : end_index ]
                entry_term_mask = entry_term_mask[ start_index : end_index ]

                # Adjust For New Token Index Of Interest
                num_subwords_before_entry_term -= start_index
                token_mask_sub_word_start_idx  -= start_index

                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -   Restricted Sequence Sub-Words: " + str( subword_tokens ) )

            # Use Sub-Word Buffer Surrounding Entry Term To Restrict Context
            #   i.e. If Sequences Have Not Be Separated By [SEP] Tokens.
            else:
                subword_buffer_size = 50

                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Restricting Term Context Using Buffer Size: " + str( subword_buffer_size ) )
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -   Sequence Sub-Words: " + str( subword_tokens ) )

                start_index     = num_subwords_before_entry_term - subword_buffer_size if num_subwords_before_entry_term - subword_buffer_size > 0 else 0
                end_index       = num_subwords_before_entry_term + entry_term_subword_length + subword_buffer_size if num_subwords_before_entry_term + entry_term_subword_length + subword_buffer_size < len( subword_tokens ) else len( subword_tokens )
                subword_tokens  = subword_tokens[ start_index : end_index ]
                entry_term_mask = entry_term_mask[ start_index : end_index ]

                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -   Restricted Sequence Sub-Words: " + str( subword_tokens ) )

            # Check Again To See If The Entry Term Mask Contains Our Desired Entry Term.
            #   If It's All '0's, Then It Doesn't. We Should Skip This Instance Since It'll Zero-Out
            #   The Network By Passing An Embedding Of All Zero Elements.
            if 1 not in entry_term_mask:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: Entry Term Mask Is All Zeros" )
                self.Print_Log( "                                                - Sequence           : " + str( text_sequence ) )
                self.Print_Log( "                                                - Preceding Sequence : " + str( preceeding_sequence ) )
                self.Print_Log( "                                                - # Preceding Splits : " + str( number_of_preceeding_splits ) )
                self.Print_Log( "                                                - Entry Term         : " + str( entry_term ) )
                self.Print_Log( "                                                - Captured Entry Term: " + str( curr_entry_term ) )
                return None
        elif restrict_context and num_subwords_before_entry_term == -1:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: 'num_subwords_before_entry_term == -1' / Skipping Context Restriction" )
        else:
            # NOTE: Assumes There Are Always Spaces Between Sub-Words, Which Isn't Always True
            # Let's Double-Check Our Entry Term Mask Matches The Actual Entry Term Sub-Words
            token_mask_sub_word_start_idx = entry_term_mask.index( 1 )
            token_mask_sub_word_length    = entry_term_mask.index( 0, token_mask_sub_word_start_idx ) - token_mask_sub_word_start_idx if entry_term_mask[-1] == 0 else len( entry_term_mask ) - 1 - token_mask_sub_word_start_idx

            temp_sub_words = subword_tokens[token_mask_sub_word_start_idx:(token_mask_sub_word_start_idx+token_mask_sub_word_length)]

            # Build Term From Extracted Sub-Word Tokens
            extracted_term = ""
            for token in temp_sub_words:
                if re.search( r'^#', token ) and token != "#":
                    extracted_term += re.sub( r'\#+', "", token )
                else:
                    extracted_term += " " + token

            # Remove Surrounding Whitespace
            extracted_term = re.sub( r'^\s+|\s+^', "", extracted_term )

            if extracted_term != entry_term:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Error: extracted_term != entry_term" )
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -        entry_term    : " + str( entry_term ) )
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -        extracted_term: " + str( extracted_term ) )
            #     return None, None
        # Encode The Entry Term Mask Over The Entire Sub-Word Sequence Containing The Entry Term Of Interest i.e. 'mask_term_sequence = True'
        #   Or Just The Entry Term Sub-Words Within The Sequence i.e. 'mask_term_sequence = False' (This Is Done Prior To Thie Segment Of Code)
        num_subwords_before_entry_term = token_mask_sub_word_start_idx if token_mask_sub_word_start_idx >= 0 else -1

        if mask_term_sequence and num_subwords_before_entry_term != -1:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Encoded Entry Term (Term Sub-Words)    : " + str( entry_term_mask ) )

            start_index, end_index = 0, 0

            # Find [SEP] Token Designating Beginning Of Sequence
            if num_subwords_before_entry_term > 0:
                sep_indices = [ idx for idx, token in enumerate( subword_tokens[ 0 : num_subwords_before_entry_term - 1 ] ) if token == "[SEP]" ]
                start_index = 0 if len( sep_indices ) < 1 else sep_indices[-1] + 1
            else:
                start_index = 0

            # Find [SEP] Token Designating End Of Sequence
            if num_subwords_before_entry_term + 1 < len( subword_tokens ):
                sep_indices = [ idx for idx, token in enumerate( subword_tokens[ num_subwords_before_entry_term + 1 : len( subword_tokens ) ] ) if token == "[SEP]" ]
                end_index   = len( subword_tokens ) if len( sep_indices ) < 1 else sep_indices[0] + num_subwords_before_entry_term
            else:
                end_index   = len( subword_tokens )

            # Set Desired Sequence Indices In Entry Term Mask To '1'
            entry_term_mask = [ 1 if idx >= start_index and idx <= end_index else 0 for idx, val in enumerate( entry_term_mask ) ]

            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - Encoded Entry Term (Sequence Sub-Words): " + str( entry_term_mask ) )
        else:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() - 'mask_term_sequence == False' / Disabled or Error: 'num_subwords_before_entry_term == -1'" )

        # Extract BERT Tokenizer Encodings:
        #   1) Sub-Word Input IDs
        #   2) Attention Masks
        #   3) Token Type IDs
        inputs = self.tokenizer.encode_plus( subword_tokens, add_special_tokens = True, max_length = self.max_sequence_length )

        # Adjust Term Sub-Word Start/End Indices (Added [CLS] Token To Beginning Of Sequence So We Shift The Term Start Index By 1)
        token_mask_sub_word_start_idx += 1

        # Add Mask Values '(0)' For Special Tokens To 'entry_term_mask'
        entry_term_mask = [0] + entry_term_mask + [0]

        encoded_text_id_sequence, attention_mask, token_type_ids = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]

        # Pad Model Input Lists
        #   Note: This Assumes 'padding_token_value' Is '0'. If Not, This Variable Should Be Changed Accordingly.
        #         BERT's Attention Mask Expects Either '1' Or '0'. The 'entry_term_mask' Expects The Same.
        #         I Could Just Change It Now... But... ¯\_(ツ)_/¯ ... Ehh
        if pad_input and self.max_sequence_length - len( encoded_text_id_sequence ) > 0:
            padding_length = self.max_sequence_length - len( encoded_text_id_sequence )
            encoded_text_id_sequence.extend( [self.sub_word_pad_token_id] * padding_length )
            attention_mask.extend( [self.sub_word_pad_token_id] * padding_length )
            token_type_ids.extend( [self.sub_word_pad_token_id] * padding_length )
            entry_term_mask.extend( [0] * padding_length )

        # Encoded Entry Term IDs List (Only Used For Debugging Purposes)
        encoded_entry_term = [ encoded_text_id_sequence[idx] for idx, mask_value in enumerate( entry_term_mask ) if mask_value == 1 ]

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Entry Term Sub-Words: " + str( self.tokenizer.convert_ids_to_tokens( encoded_entry_term ) ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Encoded Entry Term  : " + str( encoded_entry_term       ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Encoded Sequence    : " + str( encoded_text_id_sequence ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Attention Mask      : " + str( attention_mask           ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Token Type ID       : " + str( token_type_ids           ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Input_Instance() -      Entry Term Mask     : " + str( entry_term_mask          ) )

        return ( encoded_text_id_sequence, attention_mask, token_type_ids, entry_term_mask )

    """
        Encodes/Vectorized Concept Mapping/Entity Linking Instance Data

        Inputs:
            annotation_concept   : Concept ID / CUI (String)
            pad_output           : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                   i.e. Categorical Crossentropy Loss vs. Sparse Categorical Crossentropy Loss Formats
            concept_delimiter    : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                   Used For One-To-Many Relationships
            label_per_sub_word   : Produces An Output Label For Each Sub-Word Token In The Sequence (Bool)
            encoded_inputs       : Encoded Inputs - Returned Back From self.Encoded_CL_Input_Instance() Function

        Outputs:
            encoded_concept      : Candidate Concept Embedding
    """
    def Encode_CL_Output_Instance( self, annotation_concept, pad_output = False, concept_delimiter = None, label_per_sub_word = False, encoded_inputs = None ):
        # Check(s)
        if self.Get_Number_Of_Unique_Concepts() == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None

        if not encoded_inputs and label_per_sub_word:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: 'label_per_sub_word = True' And 'encoded_inputs == None'", force_print = True )
            return None

        token_ids, attention_mask, token_type_ids, entry_term_mask = None, None, None, None
        token_mask_sub_word_start_idx, token_mask_sub_word_length  = -1, -1

        if encoded_inputs:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Valid Encoded Input Provided" )

            token_ids, attention_mask, token_type_ids, entry_term_mask = encoded_inputs

            # Determine Entry Term Mask Index Boundaries
            #   'token_mask_sub_word_length' Also Accounts For Entry Term Sub-Words Ending At The End Of The Sequence
            #    i.e. 'token_mask_sub_word_length = len( entry_term_mask ) - 1 - token_mask_sub_word_start_idx
            token_mask_sub_word_start_idx = entry_term_mask.index( 1 )
            token_mask_sub_word_length    = entry_term_mask.index( 0, token_mask_sub_word_start_idx ) - token_mask_sub_word_start_idx if entry_term_mask[-1] == 0 else len( entry_term_mask ) - 1 - token_mask_sub_word_start_idx

        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Encoding Concept: " + str( annotation_concept ) )

        # ----------------- #
        # Encode Concept(s) #
        # ----------------- #

        concept_id, concept_ids, encoded_concept = None, [], []

        # Encode Concept ID Outputs Per Sub-Word Token
        if label_per_sub_word:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Label Per Sub-Word = True" )

            if concept_delimiter is not None:
                if token_mask_sub_word_start_idx == -1:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: Token Mask Sub-Word Index == -1 / Entry Term Sub-Word Not Found" )
                    return None

                self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Building Multi-Label/Multi-Concept Output" )

                # Create Multi-Label Vectors For Each Sub-Word In The Entry Term
                for concept_idx, concept in enumerate( annotation_concept.split( concept_delimiter ) ):
                    # Skip Composite Mentions If 'self.skip_composite_mentions == True'
                    if self.skip_composite_mentions and concept_idx > 0:
                        continue

                    concept_id = self.Get_Concept_ID( concept )

                    if concept_id == -1:
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( concept ) + "\' Not In Dictionary", force_print = True )
                        continue

                    concept_ids.append( concept_id )

                if len( concept_ids ) == 0:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: Vector Length Of Concept IDs == 0" )
                    return None
            else:
                if token_mask_sub_word_start_idx == -1:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: Token Mask Sub-Word Index == -1 / Entry Term Sub-Word Not Found" )
                    return None

                concept_id = self.Get_Concept_ID( annotation_concept )

                if concept_id == -1:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( annotation_concept ) + "\' Not In Dictionary", force_print = True )
                    return None

                concept_ids.append( concept_id )

            # Encode The Concept ID Output Labels
            for token_idx, val in enumerate( token_ids ):
                sub_word_concept_ids = None

                if pad_output:
                    # Create Concept Label Vector
                    concept_labels = [ 0 for _ in range( self.Get_Number_Of_Unique_Concepts() ) ]

                    # Encode All Entry Term Sub-Word Using Our Concept ID Dictionary Concept Labels
                    if token_idx >= token_mask_sub_word_start_idx and token_idx <= token_mask_sub_word_start_idx + token_mask_sub_word_length - 1:
                        if len( concept_ids ) == 1:
                            concept_labels[ concept_ids[0] ] = 1
                        else:
                            for concept_id in concept_ids: concept_labels[concept_id] = 1
                    # Encode All Other Sub-Words Outside The Entry Term As Padding Or 'O' Label
                    else:
                        concept_labels[ self.Get_Concept_ID( self.Get_Padding_Token() ) ] = 1

                    encoded_concept.append( concept_labels )
                else:
                    # Encode All Entry Term Sub-Word Using Our Concept ID Dictionary Concept Labels
                    if token_idx >= token_mask_sub_word_start_idx and token_idx <= token_mask_sub_word_start_idx + token_mask_sub_word_length - 1:
                        sub_word_concept_ids = concept_ids[0] if len( concept_ids ) == 1 else concept_ids
                    # Encode All Other Sub-Words Outside The Entry Term As Padding Or 'O' Label
                    else:
                        sub_word_concept_ids = self.Get_Concept_ID( self.Get_Padding_Token() )

                    encoded_concept.append( sub_word_concept_ids )

        # Encode Single/Multiple Concept ID Output(s) Per Instance
        else:
            self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Building Per-Instance Singule/Multi-Label Concept Label" )

            # If Padding Output, Build Concept List Of All Zero Elements
            encoded_concept = [ 0 for _ in range( self.Get_Number_Of_Unique_Concepts() ) ] if pad_output else []

            if concept_delimiter is not None:
                self.Print_Log( "BERTSimilarityDataLoader::Encode_CL_Output_Instance() - Building Multi-Label/Multi-Concept Output" )

                for concept_idx, concept in enumerate( annotation_concept.split( concept_delimiter ) ):
                    # Skip Composite Mentions If 'self.skip_composite_mentions == True'
                    if self.skip_composite_mentions and concept_idx > 0:
                        continue

                    concept_id = self.Get_Concept_ID( concept )

                    if concept_id == -1:
                        self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( concept ) + "\' Not In Dictionary", force_print = True )
                        return None

                    concept_ids.append( concept_id )
            else:
                concept_id = self.Get_Concept_ID( annotation_concept )

                if concept_id == -1:
                    self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Warning: Concept \'" + str( annotation_concept ) + "\' Not In Dictionary", force_print = True )
                    return None

                concept_ids.append( concept_id )

            if pad_output:
                for concept_id in concept_ids: encoded_concept[concept_id] = 1
            else:
                for concept_id in concept_ids: encoded_concept.append( concept_id )

            # Check To See If Any Concept IDs Were Encoded
            if pad_output and 1 not in encoded_concept:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: No Concept IDs Were Encoded For Multi-Output Instance", force_print = True )
                return None
            elif not pad_output and len( encoded_concept ) == 0:
                self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Error: No Concept IDs Were Encoded For Multi-Output Instance", force_print = True )
                return None

            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Output_Instance() - Encoded Concept: " + str( encoded_concept ) )

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
            encoded_concept      : Encoded Concept Vector (List/Vector Of Integers)
    """
    def Encode_CL_Instance( self, text_sequence, entry_term, annotation_concept, annotation_indices, pad_input = True, pad_output = False,
                            concept_delimiter = None, mask_term_sequence = False, separate_sentences = True, term_sequence_only = False,
                            restrict_context = False, label_per_sub_word = False, limit_to_max_sequence = True ):
        # Check(s)
        if len( self.concept_id_dictionary ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Instance() - Error: Concept ID Dictionary Is Empty", force_print = True )
            return None, None

        encoded_inputs = self.Encode_CL_Input_Instance( text_sequence = text_sequence, entry_term = entry_term, annotation_indices = annotation_indices,
                                                        pad_input = pad_input, mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                        term_sequence_only = term_sequence_only, restrict_context = restrict_context,
                                                        limit_to_max_sequence = limit_to_max_sequence )

        if encoded_inputs is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Instance() - Error Encoding Input" )
            return None, None

        encoded_ouput  = self.Encode_CL_Output_Instance( annotation_concept = annotation_concept, pad_output = pad_output, concept_delimiter = concept_delimiter,
                                                         label_per_sub_word = label_per_sub_word, encoded_inputs = encoded_inputs )

        if encoded_ouput is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Encode_CL_Instance() - Error Encoding Output" )
            return None, None

        return encoded_inputs, encoded_ouput

    """
        Decodes Sequence Of Token IDs to Sequence Of Token Strings

        Inputs:
            encoded_input_sequence : Sequence Of Token IDs
            remove_padding         : Removes Padding Tokens From Returned String Sequence

        Outputs:
            decoded_input_sequence : Decoded Text Sequence (List Of Strings)
    """
    def Decode_NER_Input_Instance( self, encoded_input_sequence, remove_padding = False, convert_subwords_to_tokens = False ):
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Input_Instance() - Encoded Token ID Sequence: " + str( encoded_input_sequence ) )

        # Alternative Method
        # decoded_input_instance = self.tokenizer.decode( encoded_text_sequence )

        decoded_input_sequence = self.tokenizer.convert_ids_to_tokens( encoded_input_sequence )

        if remove_padding and "[SEP]" in decoded_input_sequence:
            # Get Index Of [SEP] Token (Should Be End Of Sequence String)
            sep_index              = decoded_input_sequence.index( "[SEP]" )
            decoded_input_sequence = decoded_input_sequence[0:sep_index + 1]
        elif remove_padding and "[PAD]" in decoded_input_sequence:
            decoded_input_sequence = [token for token in decoded_input_sequence if token != "[PAD]"]

        # Old Manual Method
        token_sequence       = []
        current_token        = ""
        word_join_characters = []   # word_join_characters = [ "-" ]    # WIP

        for sub_word in decoded_input_sequence:
            # Detect If Current Sub-Word Is A Multi-Sub-Word Token
            temp_sub_word = re.sub( r'^#+', "", sub_word )

            # Only Join If Sub-Word Contains '#' Before Sub-Word String. If Sub-Word Is One-Or-More '#' Character, It Is Not A Multi-Word Token
            if convert_subwords_to_tokens and re.search( r'^#+', sub_word ) and len( temp_sub_word ) > 0 or sub_word in word_join_characters:
                current_token += re.sub( r'^#+', "", sub_word )
            else:
                # Add Previously Found Token To Token Sequence List (Non-Multi-Sub-Word Token)
                #   Or Add Multi-Sub-Word Token to Token Sequence List
                if current_token != "": token_sequence.append( current_token )

                # Set New Sub-Word To Current Token Variable
                current_token = sub_word

        # Add Remaining Text
        if current_token != "": token_sequence.append( current_token )

        decoded_input_sequence = token_sequence

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Input_Instance() - Decoded Token Sequence: " + str( decoded_input_sequence ) )

        return decoded_input_sequence

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings

        Inputs:
            encoded_output_sequence : Sequence Of Encoded Label IDs

        Outputs:
            decoded_output_sequence : Sequence Of Decoded Label IDs (List Of Text Labels)
    """
    def Decode_NER_Output_Instance( self, encoded_output_sequence ):
        # Check(s)
        if isinstance( encoded_output_sequence, list ) and len( encoded_output_sequence ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, COO        ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []

        # More Encoded Data Checks / Convert Data-Types 'csr_matrix' And 'COO' Into Numpy Arrays
        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = np.asarray( encoded_output_sequence )[0]

        if isinstance( encoded_output_sequence, COO ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        if isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Encoded Output ID Sequence: " + str( encoded_output_sequence ) )

        idx_to_tokens = { v:k for k,v in self.Get_Annotation_Labels().items() }

        # Convert The Predicted Entity Indices To Entity Tokens
        if encoded_output_sequence.ndim == 1:   # Sparse Categorical Crossentropy
            decoded_output_sequence = [ idx_to_tokens[prediction_index] for prediction_index in encoded_output_sequence ]
        elif encoded_output_sequence.ndim == 2: # Binary / Categorical Crossentropy
            decoded_output_sequence = [ [ idx_to_tokens[np.argmax( token_instance )] for token_instance in encoded_output_sequence ] ]
            if np.ndim( decoded_output_sequence ) == 2: decoded_output_sequence = decoded_output_sequence[0]

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Decoded Output Sequence: " + str( decoded_output_sequence ) )

        return decoded_output_sequence

    """
        Decodes Input & Output Sequence Of NER Token IDs And Label IDs To Sequence Of NER Token & Label Strings

        Inputs:
            encoded_input_sequence    : Sequence Of Encoded Token IDs
            encoded_output_sequence   : Sequence Of Encoded Label IDs
            remove_padding            : Removes [PAD] Token From Decoded Input Sequence (True/False)
            remove_special_characters : Removed [CLS] and [SEP] Tokens From Decoded Input/Output Sequence (True/False)

        Outputs:
            decoded_input_sequence    : Sequence Of Decoded Token IDs (List Of Text Tokens)
            decoded_output_sequence   : Sequence Of Decoded Label IDs (List Of Text Labels)
    """
    def Decode_NER_Instance( self, encoded_input_sequence, encoded_output_sequence, remove_padding = True, remove_special_characters = False ):
        # Check(s)
        if isinstance( encoded_input_sequence, list ) and len( encoded_input_sequence ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Error: Encoded Input Sequence Length == 0" )
            return []
        if isinstance( encoded_input_sequence, np.ndarray ) and encoded_input_sequence.shape[0] == 0 or \
           isinstance( encoded_input_sequence, COO        ) and encoded_input_sequence.shape[0] == 0 or \
           isinstance( encoded_input_sequence, csr_matrix ) and encoded_input_sequence.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Error: Encoded Input Sequence Length == 0" )
            return []
        if isinstance( encoded_output_sequence, list ) and len( encoded_output_sequence ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Error: Encoded Output Sequence Length == 0" )
            return []
        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, COO        ) and encoded_output_sequence.shape[0] == 0 or \
           isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Error: Encoded Output Sequence Length == 0" )
            return []

        # Encoded Input Checks / Convert Data-Types 'csr_matrix' And 'COO' Into Numpy Arrays
        if isinstance( encoded_input_sequence, np.ndarray ) and encoded_input_sequence.ndim == 3:
            encoded_input_sequence = np.asarray( encoded_input_sequence )[0]

        if isinstance( encoded_input_sequence, COO ) and encoded_input_sequence.ndim == 3:
            encoded_input_sequence = encoded_input_sequence.todense()[0]

        if isinstance( encoded_input_sequence, csr_matrix ) and encoded_input_sequence.ndim == 3:
            encoded_input_sequence = encoded_input_sequence.todense()[0]

        # Encoded Output Checks / Convert Data-Types 'csr_matrix' And 'COO' Into Numpy Arrays
        if isinstance( encoded_output_sequence, np.ndarray ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = np.asarray( encoded_output_sequence )[0]

        if isinstance( encoded_output_sequence, COO ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        if isinstance( encoded_output_sequence, csr_matrix ) and encoded_output_sequence.ndim == 3:
            encoded_output_sequence = encoded_output_sequence.todense()[0]

        # Final Check(s)
        if len( encoded_input_sequence ) != len( encoded_output_sequence ):
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Error: Encoded Input Sequence Length != Encoded Output Sequence Length", force_print = True )
            return [], []

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Encoded Input Token ID Sequence : " + str( encoded_input_sequence  ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Encoded Output Token ID Sequence: " + str( encoded_output_sequence ) )

        decoded_input_sequence, decoded_output_sequence = [], []

        # Decode Input Sequence
        decoded_input_sequence = self.tokenizer.convert_ids_to_tokens( encoded_input_sequence )

        # Remove Padding From Encoded Input Sequence
        if remove_padding:
            # Get Index Of [SEP] Token (Should Be End Of Sequence String)
            sep_index               = decoded_input_sequence.index( "[SEP]" )
            decoded_input_sequence  = decoded_input_sequence[0:sep_index + 1]
            encoded_output_sequence = encoded_output_sequence[0:sep_index + 1]

        # Decode Input Sequence Output Labels
        #   Convert The Predicted Entity Indices To Entity Tokens
        idx_to_tokens           = { v:k for k,v in self.Get_Annotation_Labels().items() }
        decoded_output_sequence = [ idx_to_tokens[prediction_index] for prediction_index in encoded_output_sequence ]

        token_sequence = []
        label_sequence = []
        current_token  = ""
        current_label  = ""
        word_join_characters = []

        # First Sub-Word Label Determines The Token Label
        for idx, sub_word in enumerate( decoded_input_sequence ):
            # Detect If Current Sub-Word Is A Multi-Sub-Word Token
            if re.search( r'^#+', sub_word ) or sub_word in word_join_characters:
                current_token += re.sub( r'^#+', "", sub_word )
            else:
                # Add Previously Found Token To Token Sequence List (Non-Multi-Sub-Word Token)
                #   Or Add Multi-Sub-Word Token to Token Sequence List
                if current_token != "":
                    token_sequence.append( current_token )
                    label_sequence.append( current_label )

                    current_token = ""
                    current_label = ""

                # Set New Sub-Word Label To Current Label
                if current_token == "": current_label = decoded_output_sequence[idx]

                # Set New Sub-Word To Current Token Variable
                current_token = sub_word

        # Add Remaining Text
        if current_token != "":
            token_sequence.append( current_token )
            label_sequence.append( current_label )

        if remove_special_characters:
            if token_sequence[0] == "[CLS]":
                token_sequence = token_sequence[1:]
                label_sequence = label_sequence[1:]
            if token_sequence[-1] == "[SEP]":
                token_sequence = token_sequence[0:-1]
                label_sequence = label_sequence[0:-1]

        decoded_input_sequence  = token_sequence
        decoded_output_sequence = label_sequence

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Decoded Input Token Sequence : " + str( decoded_input_sequence  ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Decoded Output Token Sequence: " + str( decoded_output_sequence ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Instance() - Complete" )

        return decoded_input_sequence, decoded_output_sequence

    """
        Decodes Input Sequence Instance Of IDs To Entry Term String(s)

        Note: This Assume Only One Entry Term Exists Per Input Sequence.

        Inputs:
            encoded_input_instance : Input Sequence Of Sub-Word IDs (List/Numpy Array)
            entry_term_mask        : Entry Term Mask (List/Numpy Array Of 0's and 1's)

        Outputs:
            decoded_entry_term     : Decoded Entry Term (String)
    """
    def Decode_CL_Input_Instance( self, encoded_input_instance, entry_term_mask ):
        # Check(s)
        if entry_term_mask is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Input_Instance() - Error: Function Requires Valid 'entry_term_mask' / 'None' Provided", force_print = True )
            return ""

        if not isinstance( encoded_input_instance, np.ndarray ): encoded_input_instance = np.asarray( encoded_input_instance )
        if not isinstance( entry_term_mask       , np.ndarray ): entry_term_mask        = np.asarray( entry_term_mask )

        if encoded_input_instance.shape != entry_term_mask.shape:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Input_Instance() - Error: 'encoded_input_instance.shape != entry_term_mask.shape / Unable To Decode", force_print = True )
            return ""

        decoded_entry_term = []
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Input_Instance() - Decoding Output Label Instance: " + str( encoded_input_instance ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Input_Instance() -                Entry Term Mask: " + str( entry_term_mask ) )

        # First We Apply The Entry Term Mask Over The Input Sequence Of Sub-Word IDs
        encoded_input_instance *= entry_term_mask

        # Now We Extract Which Sub-Word IDs Aren't Zero'ed Out To Determine Out Entry Term(s)
        encoded_input_instance = [ val for val in encoded_input_instance if val != 0 ]

        # Next We Decode The Sub-Word IDs Into Sub-Word Strings
        decoded_sub_words = self.tokenizer.convert_ids_to_tokens( encoded_input_instance )

        # Check For Multi-Term Token i.e. Sub-Word Token Without '###' Prefix
        token_concat_list = [ "-", ",", "[", "]", "(", ")", "{", "}", "/", ".", "+", "\\", "'", "`", "%", ":" ]
        decoded_sub_words = [ "[<S>]" + sub_word if idx > 0
                              and sub_word not in token_concat_list
                              and decoded_sub_words[idx-1] not in token_concat_list
                              and not re.search( r'^#', sub_word ) else sub_word
                              for idx, sub_word in enumerate( decoded_sub_words ) ]

        # Finally, We Combine The Sub-Words Into A Single Token
        decoded_entry_term = [ re.sub( r'^#+', "", sub_word ) for sub_word in decoded_sub_words ]
        decoded_entry_term = "".join( decoded_entry_term ) if len( decoded_sub_words ) > 0 else ""

        # Replace Our Special Space Character Identifer With Whitespace To Designate A Multi-Word Entry Term
        decoded_entry_term = re.sub( r'\[\<S\>\]', " ", decoded_entry_term )

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Input_Instance() - Decoding Output Label Instance: " + str( decoded_entry_term ) )

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
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
            return []
        if isinstance( encoded_output_labels, np.ndarray ) and encoded_output_labels.shape[0] == 0 or \
           isinstance( encoded_output_labels, COO        ) and encoded_output_labels.shape[0] == 0 or \
           isinstance( encoded_output_labels, csr_matrix ) and encoded_output_labels.shape[0] == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Decode_NER_Output_Instance() - Error: Encoded Sequence Length == 0" )
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

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Output_Instance() - Decoding Output Label Instance: " + str( encoded_output_labels ) )

        for idx, val in enumerate( encoded_output_labels ):
            # Perform Thresholding
            #   Note - This Is Only Applicable When Utilizing 'Sigmoid' As The Final Activation Function.
            #          As 0.5 Is The Inflection Point Of The Function.
            if val > 0.5:
                # Fetch Decoded Concept ID String Using Index
                decoded_output = self.Get_Concept_From_ID( idx )

                # Append Concept ID String To Decoded Concept ID List
                if decoded_output not in decoded_output_labels: decoded_output_labels.append( decoded_output )

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Output_Instance() - Decoded Output Label Instance: " + str( decoded_output_labels ) )

        return decoded_output_labels

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self, encoded_input_instance, entry_term_mask, encoded_output_labels ):
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Instance() - Encoded Sequence     : " + str( encoded_input_instance ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Instance() - Entry Term Mask      : " + str( entry_term_mask ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Instance() - Encoded Output Labels: " + str( encoded_output_labels ) )

        decoded_entry_term    = self.Decode_CL_Input_Instance( encoded_input_instance = encoded_input_instance, entry_term_mask = entry_term_mask )
        decoded_output_labels = self.Decode_CL_Output_Instance( encoded_output_labels = encoded_output_labels )

        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Instance() - Decoded Entry Term   : " + str( decoded_entry_term ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Decode_CL_Instance() - Decoded Output Labels: " + str( decoded_output_labels ) )

        return decoded_entry_term, decoded_output_labels

    """
        Generates List Of Text Sequence Splits Given A Maximum Sub-Word Sequence Size
           i.e. Split Sequence Into Sequences Of Sub-Words For BERT Model.

        Example: Sequence Containing 110 Tokens Will Be Split Into Sub-Words And Packed
                 Into Sub-Strings Of 512 Sub-Word Length.

        Inputs:
            passage             : Passage Object Containing Text Sequence, Annotations, Indices, Entity Types, etc. (Passage Object)
            max_sequence_length : Maximum Sequence Length (Integer)

        Outputs:
            sequence_index_list : List Of Sequence Start & End Indices (List)
    """
    def Get_Sequence_Splits( self, passage, max_sequence_length = 128 ):
        # ------------------------------- #
        # Build Annotation-Per-Token List #
        # ------------------------------- #

        text_sequence = passage.Get_Passage()

        text_sequence_character_mask = [ "_" if text_sequence[i] != " " else " " for i in range( len( text_sequence ) ) ]

        for annotation, annotation_label, annotation_idx, is_composite_mention, is_individual_mention in zip( passage.Get_Annotations(), passage.Get_Annotation_Labels(),
                                                                                                              passage.Get_Annotation_Indices(), passage.Get_Composite_Mention_List(),
                                                                                                              passage.Get_Individual_Mention_List() ):
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
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() - Error: Extracted Token != True Token",              force_print = True )
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() -        True Token     : " + str( annotation ),      force_print = True )
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() -        Extracted Token: " + str( extracted_token ), force_print = True )
                elif number_of_indices > 1 and extracted_token not in annotation:
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() - Error: Extracted Token Not In True Token",          force_print = True )
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() -        True Token     : " + str( annotation ),      force_print = True )
                    self.Print_Log( "BERTBioCreativeDataLoader::Get_Sequence_Splits() -        Extracted Token: " + str( extracted_token ), force_print = True )

                text_sequence_character_mask[annotation_offset:annotation_end]  = ["#"] * ( annotation_end - annotation_offset )

                # Double Check To See If The Annotation Doesn't Contain Spaces (Annotation Not Composed Of Multiple Singular Terms)
                space_indices = [i for i, char in enumerate( annotation ) if char == " "]

                # Fill Spaces Back In
                for index in space_indices:
                    offset_index = annotation_offset + index
                    text_sequence_character_mask[offset_index]     = " "
                    text_sequence_character_mask[offset_index + 1] = "#"

        text_sequence_character_mask  = '' . join( text_sequence_character_mask )
        text_sequence_annotation_list = text_sequence_character_mask.split()

        # --------------------------------------- #
        # Extract Indicies Of Max Sequence Length #
        # --------------------------------------- #

        current_index                       = 0
        text_sequence_list                  = passage.Get_Passage().split()
        passage_list_length                 = len( text_sequence_list )
        sequence_index_list, subword_tokens = [], []
        temp_lists                          = []

        # Tokenize Data Into Sub-Words And Assign Label IDs Per Sub-Word
        while True:
            for ( index, token ), annotation_token in zip( enumerate( text_sequence_list ), text_sequence_annotation_list ):
                token_subwords = self.tokenizer.tokenize( token )

                # Check For End Of Text Sequence Reached
                if current_index + index >= passage_list_length - 1:
                    text_sequence_list            = text_sequence_list[index+1:]
                    text_sequence_annotation_list = text_sequence_annotation_list[index+1:]
                    sequence_index_list.append( str( current_index ) + ":" + str( current_index + index + 1 ) )
                    current_index += index + 1
                    temp_lists.append( subword_tokens + token_subwords )
                    token_subwords = []

                # Keep Adding Sub-Words To Current Sub-Word List Until We've Reached Max Sequence Length
                if len( subword_tokens ) + len( token_subwords ) < max_sequence_length:
                    subword_tokens.extend( token_subwords )
                # Given Passage, Declare Sub-String Start And End Indices Of Max Sequence Length
                else:
                    # Remove Tokens From 'text_sequence_list'
                    if "#" in annotation_token: index -= 1
                    text_sequence_list            = text_sequence_list[index+1:]
                    text_sequence_annotation_list = text_sequence_annotation_list[index+1:]
                    sequence_index_list.append( str( current_index ) + ":" + str( current_index + index + 1 ) )
                    current_index += index + 1
                    temp_lists.append( subword_tokens )
                    subword_tokens = []
                    break

            if current_index >= passage_list_length - 1 or len( text_sequence_list ) == 0: break

        return sequence_index_list



    """
        Combines Similar Input Instances Into A Single Instance. Merges Entry Term Masks And Output Concept ID Labels.

        Inputs:
            encoded_sequences        : Encoded Concept Linking Input Instances / Sequences Of Tokenized Sub-Word IDs (List)
            term_masks               : Encoded Entry Masks (List)
            concept_id_labels        : Concept ID Labels (List)

        Outputs:
            merged_encoded_sequences : Merged Encoded Concept Linking Input Instances / Sequences Of Tokenized Sub-Word IDs (List)
            merged_term_masks        : Merged Encoded Entry Masks (List)
            merged_concept_id_labels : Merged Concept ID Labels (List)
    """
    def Merge_Concept_Linking_Instances( self, encoded_sequences, entry_term_masks, concept_id_labels ):
        # Storage Variables
        merged_encoded_sequences, merged_term_masks, merged_concept_id_labels = [], [], []

        # For Each Encoded Input Instance, Search The Encoded List/Array For The Indices Of Similar Input Instances.
        #   We Then Want To Combine Their Asosciated Entry Term Masks And Output Concept ID Labels To Form A Single Instance.
        #   Since Each Instance Can Correspond To Many Concept Labels, We're Going To Create An Indexing Scheme Using The
        #     Entry Term Mask. Then Concert The 'concept_id_labels' To A List Which Relates The Sub-Words To The Correct
        #     Entry Terms By Index.
        for curr_idx, ( encoded_sequence, entry_term_mask, concept_ids ) in enumerate( zip( encoded_sequences, entry_term_masks, concept_id_labels ) ):
            encoded_sequence = list( encoded_sequence )
            concept_ids      = list( concept_ids )

            # Skip Merged Sequences Already In List
            if encoded_sequence in merged_encoded_sequences: continue

            # Fetch Remaining Input Instance Sequences Which Are Similar To The Current Sequence
            similar_instance_indices = [ idx for idx, sequence in enumerate( encoded_sequences ) if idx != curr_idx and np.array_equal( sequence, encoded_sequence ) ]

            # We Found More Than One Of The Same Input Instance, So Let's Merge Them
            if len( similar_instance_indices ) > 0:
                # We Don't Really Need To Merge The Encoded Input Sequence Since They're The Same.
                #   Just The Entry Term Masks And Concept ID Labels.
                merged_encoded_sequences.append( encoded_sequence )

                # Create Storage For Our Concept IDs
                temp_concept_ids = [ "<PAD>", concept_ids ]

                # Merge The Entry Term Masks And Concept IDs
                #   NOTE: Remaining Array Elements Should Not Exceed '1' Are Summing.
                #         If So, Something Went Wrong During Pre-Processing.
                modifier_value = 1

                for idx in similar_instance_indices:
                    # Only Modify If Mask Array Contains One Or More Elements > 0
                    if np.amax( entry_term_masks[idx], axis = 0 ) > 0:
                        temp_entry_term_mask = entry_term_masks[idx]

                        # Modify The Term Mask By A Value To Create Our Distinct Indices
                        temp_entry_term_mask = [ val + modifier_value if val != 0 else 0 for val in temp_entry_term_mask ]

                        # Add The Modified Entry Term Mask To Our Current One
                        entry_term_mask += temp_entry_term_mask

                        # Merge Our Remaining Concept IDs
                        temp_concept_ids.append( list( concept_id_labels[idx] ) )

                        # Increment The Mask Modifier Value
                        modifier_value += 1

                merged_term_masks.append( list( entry_term_mask ) )
                merged_concept_id_labels.append( temp_concept_ids )
            else:
                merged_encoded_sequences.append( encoded_sequence )
                merged_term_masks.append( list( entry_term_mask ) )
                merged_concept_id_labels.append( [ "<PAD>", concept_ids ] )

        merged_encoded_sequences = np.asarray( merged_encoded_sequences )
        merged_term_masks        = np.asarray( merged_term_masks )

        # Return Back Merged Lists:
        #       'merged_encoded_sequences': Numpy Array Of Encoded Input Ids
        #       'merged_term_masks        : Numpy Array Of Merged Term Masks
        #       'merged_concept_id_labels': List Of List Where Index Corresponds To Values Within 'merged_term_masks'
        return merged_encoded_sequences, merged_term_masks, merged_concept_id_labels

    """
        Reads Original Data File And Compares Original Passage (Text) Sequences To The Pre-Processed Counterparts For Each Document By
          Their Associated Document ID. Pre-Processed Instance Sequence Data Is Converted To The Un-Processed Form By Aligning The BERT
          Tokenizer Pre-Processed Sequence Data. This Also Keep Track Of Entity Token Labels And Exact Span Indices Of Each Token Per
          Entity Label.

        NOTE: This Function Is Dependent On The Evaluation File. It Reads All Documents/Passages While Mapping Each Passage Instance
              To The Pre-Processed Instance And Re-populating The Evaluation Document Passage Data To Create The Formatted Output File.

              This May Be Fixed To Not Require The Evaluation File In The Future.

        Inputs:
            read_file_path            : Original Evaluation File To Read From (String)
            write_file_path           : Write File Path / Path To Write BioC Formatted Output Data (String)
            data_list                 : List Of Passage Objects, Parsed From BioC XML/JSON File (List Of Passage Objects)
            encoded_ner_inputs        : List Of BERT Tokenizer Encoded NER Input Instances (List Of Token IDs)
            encoded_ner_outputs       : List Of BERT Tokenizer Encoded NER Output Instances (List Of Label IDs)
            encoded_concept_inputs    : List Of Encoded Concept Input Instances (List Of Token IDs)
            encoded_concept_outputs   : List Of Encoded Concept Output Instances (List Of Label IDs)

        Outputs:
            None
    """
    def Write_Formatted_File( self, read_file_path = "", write_file_path = "bioc_formatted_file.xml", data_list = [],
                              encoded_ner_inputs = None, encoded_ner_outputs = None, encoded_concept_inputs = None, encoded_concept_outputs = None ):
        # Check(s)
        if encoded_ner_inputs is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded NER Inputs == None" )
        if encoded_ner_outputs is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded NER Outputs == None" )
        if encoded_concept_inputs is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded Concept Inputs == None" )
        if encoded_concept_outputs is None:
            self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Warning: Encoded Concept Outputs == None" )

        # Concept Output Data-Type Check
        if isinstance( encoded_concept_outputs, list ): encoded_concept_outputs = np.asarray( encoded_concept_outputs )

        # Use Data Instances Passed By Parameter, If 'data_list = None' Use Internally Stored Instances
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Warning: 'data_list' Is Empty / Using Data List Stored In Memory" )
            data_list = self.data_list

        # Storage Lists For NER & CL
        encoded_ner_sequences, encoded_ner_labels, encoded_cl_sequences, encoded_cl_labels, encoded_cl_term_masks = [], [], [], [], []

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

        # Extract The Encoded Sequence IDs And Term Masks For CL
        if encoded_concept_inputs is not None and isinstance( encoded_concept_inputs, tuple ):
            encoded_term_masks     = encoded_concept_inputs[-1]
            encoded_concept_inputs = encoded_concept_inputs[0]

            # Merge Concept Linking Instances - Assumes Encoded Concept Outputs Is Also Not 'None'
            encoded_concept_inputs, encoded_term_masks, encoded_concept_outputs = self.Merge_Concept_Linking_Instances( encoded_sequences = encoded_concept_inputs,
                                                                                                                        entry_term_masks  = encoded_term_masks,
                                                                                                                        concept_id_labels = encoded_concept_outputs )

            # Map Encoded Concept Instance To Data List Index
            used_cl_passage_indices = list( set( self.concept_instance_data_idx ) )
            cl_data_list_to_encoded_input_map = { idx : used_cl_passage_indices.index( idx ) if idx in used_cl_passage_indices else -1 for idx in range( len( data_list ) ) }

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
                    if len( cl_data_list_to_encoded_input_map ) > 0 and cl_data_list_to_encoded_input_map[index_position] != -1 or \
                       len( cl_data_list_to_encoded_input_map ) == 0:
                        document_passage_indices.append( index_position )
                    index_position += 1
                    if index_position >= len( document_identifier_list ): break
                except ValueError as e:
                    break

            # Proceed To Convert The Pre-Processed Sequences Of Text To Their Original Un-Processed Form
            if len( document_passage_indices ) > 0:
                # If The Read Document Exists Within The Parsed File, Let's Fetch The Matching
                #   Encoded Sequences & Labels For Both NER & Concept Linking
                if encoded_ner_inputs is not None and encoded_ner_outputs is not None:
                    encoded_ner_sequences = np.asarray( encoded_ner_inputs[document_passage_indices] )
                    encoded_ner_labels    = np.asarray( encoded_ner_outputs[document_passage_indices] )

                if encoded_concept_inputs is not None and encoded_concept_outputs is not None:
                    desired_cl_indices    = [cl_data_list_to_encoded_input_map[idx] for idx in document_passage_indices if cl_data_list_to_encoded_input_map[idx] != -1]
                    encoded_cl_sequences  = np.asarray( encoded_concept_inputs[desired_cl_indices] )
                    encoded_cl_term_masks = np.asarray( encoded_term_masks[desired_cl_indices] )
                    encoded_cl_labels     = [ encoded_concept_outputs[idx] for idx in desired_cl_indices ]

                # Now Let's Decode Those Instances Back To Their Original Form, Exactly As Found In The Data File
                orig_sequences = [ data_list[idx].Get_Passage_Original() for idx in document_passage_indices ]

                # Storage Lists For NER & CL
                decoded_ner_sequences, decoded_ner_labels, decoded_cl_sequences, decoded_cl_labels, cl_term_masks = [], [], [], [], []

                # Decode NER Sequences And Labels
                #   Iterate Through NER Encoded Input & Output Instances And Decode Them
                for encoded_ner_sequence_instance, encoded_ner_label_instance in zip( encoded_ner_sequences, encoded_ner_labels ):
                    # Decode The Input/Output Instance Into A Sequence Of Text Tokens In Addition To The Entity Labels Per Token
                    decoded_ner_sequence_instance, decoded_ner_label_instance = self.Decode_NER_Instance( encoded_ner_sequence_instance, encoded_ner_label_instance,
                                                                                                          remove_padding = True, remove_special_characters = True )
                    decoded_ner_sequences.append( decoded_ner_sequence_instance )
                    decoded_ner_labels.append( decoded_ner_label_instance )

                # Decode CL Sequences And Labels
                #   Iterate Through Concept Linking Encoded Input & Output Instances And Decode Them
                for encoded_cl_sequence_instance, cl_instance_term_mask, encoded_cl_label_instances in zip( encoded_cl_sequences, encoded_cl_term_masks, encoded_cl_labels ):
                    # Decode The Input/Output Instance Into A Sequence Of Text Tokens In Addition To The Entity Labels Per Token
                    #   We're Just Using The Function Below To Convert Idx Value Lists To Sub-Word Sequences
                    decoded_cl_sequence_instance = self.Decode_NER_Input_Instance( encoded_cl_sequence_instance, remove_padding = False, convert_subwords_to_tokens = False )
                    decoded_cl_label_instance    = [ [ concept_id_list[idx] for idx, value in enumerate( instance_labels ) if value == 1.0 ] for instance_labels in encoded_cl_label_instances ]
                    cl_instance_term_mask        = list( cl_instance_term_mask )

                    # Remove [CLS], [SEP] & [PAD] Tokens From Sequence
                    if "[SEP]" in decoded_cl_sequence_instance:
                        start_index = 1 if "[CLS]" in decoded_cl_sequence_instance else 0
                        sep_index   = decoded_cl_sequence_instance.index( "[SEP]" )
                        decoded_cl_sequence_instance = decoded_cl_sequence_instance[start_index:sep_index]
                        cl_instance_term_mask        = cl_instance_term_mask[start_index:sep_index]

                    decoded_cl_sequences.append( decoded_cl_sequence_instance )
                    decoded_cl_labels.append( decoded_cl_label_instance )
                    cl_term_masks.append( cl_instance_term_mask )

                # NER/CL - Iterate Through Each Passage In The Given Document
                for passage in document.passages:
                    # Clear The Previous Sequence Annotations
                    passage.annotations.clear()

                    # Check / Skip Current Passage If Not Found Within Our Original Annotations List
                    if passage.text not in orig_sequences: continue

                    passage_offset = int( passage.offset )
                    original_text  = passage.text
                    passage_text   = self.Clean_Text( passage.text.lower() ) if self.lowercase_text else self.Clean_Text( passage.text )

                    # Check For Sequences Containing No Data Or Only Containing Whitespace
                    if len( passage_text ) == 0 or len( passage_text.split() ) == 0: continue

                    # Determine Which Of The Sequences Within The Document If The Current Passage By Index
                    passage_text_index = orig_sequences.index( original_text )

                    if len( cl_data_list_to_encoded_input_map ) > 0 and cl_data_list_to_encoded_input_map[document_passage_indices[passage_text_index]] == -1:
                        continue

                    # Determine Where The Pre-Processed Tokens Exist Within Each Original Sequence Span
                    #   Also Return The Exact Token Indices For Each Label After Returning To The Original Sequence
                    ner_seq_list, ner_label_list, ner_idx_list  = [], [], []
                    cl_seq_list, cl_term_mask_list, cl_idx_list = [], [], []

                    # NER
                    if len( decoded_ner_sequences ) > 0 and len( decoded_ner_labels ) > 0:
                        ner_seq_list, ner_label_list, ner_idx_list = self.Join_NER_Sequence_Tokens_To_Original_Form( orig_sequences[passage_text_index],
                                                                                                                     decoded_ner_sequences[passage_text_index],
                                                                                                                     decoded_ner_labels[passage_text_index] )

                    # Concept Linking (CL)
                    if len( decoded_cl_sequences ) > 0 and len( decoded_cl_labels ) > 0:
                        cl_seq_list, cl_term_mask_list, cl_idx_list, _ = self.Combine_CL_Sequence_Sub_Words_To_Original_Form( orig_sequences[passage_text_index],
                                                                                                                              decoded_cl_sequences[passage_text_index],
                                                                                                                              cl_term_masks[passage_text_index] )


                    # Write BioC Annotation Data For Both NER + CL
                    #   NOTE: 'ner_seq_list' and 'cl_seq_list' Should Contain The Same Elements. i.e. 'ner_seq_list' == 'cl_seq_list'
                    #   TODO: Account For 'ner_seq_list' != 'cl_seq_list'. (This Should Not Be Likely.)
                    if len( ner_seq_list ) > 0 and len( cl_seq_list ) > 0:
                        for idx, ( token, label, indices ) in enumerate( zip( ner_seq_list, ner_label_list, ner_idx_list ) ):
                            # Skip 'O' Entity Labels
                            if label == "O" or not re.search( r'^B-|^I-', label ): continue

                            indices              = indices.split( ":" )
                            annotation_start_idx = passage_offset + int( indices[0] )
                            annotation_length    = int( indices[1] ) - int( indices[0] )

                            # Create New BioC Annotation Object Instance And Store Appropriate Information
                            annotation           = BioCAnnotation()
                            annotation.id        = str( annotation_id )
                            annotation.text      = str( token[0:annotation_length] )    # Only Include Token Length As The Actual Annotation
                            annotation.infons["type"] = "Chemical"

                            # Insert MESH ID For Concept Linking If It Exists For A Given Token Or Nothing ("-" Character)
                            entry_term_mask_idx  = cl_term_mask_list[idx]

                            concept_id_labels = "-"

                            # Format Concept ID Labels For Passage Instance
                            if entry_term_mask_idx > 0 and len( decoded_cl_labels[passage_text_index][entry_term_mask_idx] ) > 0:
                                concept_id_labels = ",".join( decoded_cl_labels[passage_text_index][entry_term_mask_idx] )

                            # Assign The Concept ID Labels For The Given Annotation Instance (Entry Term)
                            annotation.infons["identifier"] = concept_id_labels

                            # Create New BioCLocation Object Instance For BioC Annotation Object Instance
                            #   And Store Location Information.
                            location = BioCLocation( offset = annotation_start_idx, length = annotation_length )
                            annotation.locations.append( location )
                            passage.annotations.append( annotation )

                            # Increment The Annotation ID Counter
                            annotation_id += 1

                    # Write BioC Annotation Data For NER
                    elif len( ner_seq_list ) and len( cl_seq_list ) == 0:
                        for token, label, indices in zip( ner_seq_list, ner_label_list, ner_idx_list ):
                            # Skip 'O' Entity Labels
                            if label == "O" or not re.search( r'^B-|^I-', label ): continue

                            indices              = indices.split( ":" )
                            annotation_start_idx = passage_offset + int( indices[0] )
                            annotation_length    = int( indices[1] ) - int( indices[0] )

                            # Create New BioC Annotation Object Instance And Store Appropriate Information
                            annotation           = BioCAnnotation()
                            annotation.id        = str( annotation_id )
                            annotation.text      = str( token[0:annotation_length] )    # Only Include Token Length As The Actual Annotation
                            annotation.infons["type"]       = "Chemical"
                            annotation.infons["identifier"] = "MESH:dxxxxxxx"    # Insert MESH ID For Concept Linking

                            # Create New BioCLocation Object Instance For BioC Annotation Object Instance
                            #   And Store Location Information.
                            location = BioCLocation( offset = annotation_start_idx, length = annotation_length )
                            annotation.locations.append( location )
                            passage.annotations.append( annotation )

                            # Increment The Annotation ID Counter
                            annotation_id += 1

                    # Write BioC Annotation Data For CL
                    elif len( ner_seq_list ) == 0 and len( cl_seq_list ) > 0:
                        for idx, ( token, mask_value, indices ) in enumerate( zip( cl_seq_list, cl_term_mask_list, cl_idx_list ) ):
                            # Skip '0' Mask Term Values i.e. Not Our Chemical Term Of Interest
                            if mask_value == 0: continue

                            indices              = indices.split( ":" )
                            annotation_start_idx = passage_offset + int( indices[0] )
                            annotation_length    = int( indices[1] ) - int( indices[0] )

                            # Create New BioC Annotation Object Instance And Store Appropriate Information
                            annotation           = BioCAnnotation()
                            annotation.id        = str( annotation_id )
                            annotation.text      = str( token[0:annotation_length] )    # Only Include Token Length As The Actual Annotation
                            annotation.infons["type"] = "Chemical"

                            # Insert MESH ID For Concept Linking If It Exists For A Given Token Or Nothing ("-" Character)
                            entry_term_mask_idx  = cl_term_mask_list[idx]

                            concept_id_labels = "-"

                            # Format Concept ID Labels For Passage Instance
                            if entry_term_mask_idx > 0 and len( decoded_cl_labels[passage_text_index][entry_term_mask_idx] ) > 0:
                                concept_id_labels = ",".join( decoded_cl_labels[passage_text_index][entry_term_mask_idx] )

                            # Assign The Concept ID Labels For The Given Annotation Instance (Entry Term)
                            annotation.infons["identifier"] = concept_id_labels


                            # Create New BioCLocation Object Instance For BioC Annotation Object Instance
                            #   And Store Location Information.
                            location = BioCLocation( offset = annotation_start_idx, length = annotation_length )
                            annotation.locations.append( location )
                            passage.annotations.append( annotation )

                            # Increment The Annotation ID Counter
                            annotation_id += 1

                writer.write_document( document )

        # Close BioC XML Writer
        writer.close()

        self.Print_Log( "BERTBioCreativeDataLoader::Write_Formatted_File() - Complete" )
        return True

    """
        BERT De-Tokenization

        Joins Sub-Words Back Into A Single Word/Term

            Example: ['F', '##amo', '##ti', '##dine'] -> 'Famiotidine'

        Inputs:
            tokenized_term   : List Of Term Sub-Words (List)

        Outputs:
            detokenized_term : Detokenized Word/Term (String)
    """
    def Convert_Sub_Words_To_Term( self, tokenized_term ):
        # Check(s)
        if not tokenized_term:
            self.Print_Log( "BERTBioCreativeDataLoader::Convert_Sub_Words_To_Term() - Error: 'tokenized_term' Variable Not Initialized", force_print = True )
            return ""
        elif len( tokenized_term ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Convert_Sub_Words_To_Term() - Error: 'tokenized_term' List Is Empty", force_print = True )
            return ""

        detokenized_term = ""

        for sub_word in tokenized_term:
            # Remove '#' Character
            cleaned_sub_word = re.sub( r'\#', "", sub_word )

            # Beginning Of Word Or Term (Assumes Actual Word)
            if not re.match( r'^#', sub_word ) and len( sub_word ) > 0:
                # Beginning Of A New Word In The Term (Add Whitespace)
                if len( detokenized_term ) > 0:
                    detokenized_term += " " + str( sub_word )
                # Beginning Of Term
                else:
                    detokenized_term += str( sub_word )
            # Current Token Is One Or More '#' Characters And Nothing Else
            elif re.match( r'^#', sub_word ) and len( cleaned_sub_word ) == 0:
                detokenized_term += " " + str( sub_word )
            # Subsequent Sub-Word In Word/Term
            else:
                detokenized_term += str( cleaned_sub_word )

            # Remove Surrounding Whitespace
            detokenized_term = re.sub( r'^\s+|\s+$', "", detokenized_term )

        return str( detokenized_term )

    """
        BERT NER

        Compares BERT Tokenized Sequence Tokens To The Original Non-Processed Sequence.
          Returns The Indices For Each Token Within The Sequence Along With The Token Output Labels.
          i.e. Return Pre-Processed Tokens Back To The Original Representation Given Their Originating Sequence.

        Inputs:
            original_sequence : Non-Processed Sequence Of Text (List)
            sequence_tokens   : Pre-Processed (Original) Tokenized Sequence (List)
            sequence_labels   : Token Entity Labels (List)

        Outputs:
            new_sequence      : New Sequence Of Tokens (List)
            new_labels        : Labels Associated With Tokens Within 'new_sequence' (List)
            new_idx           : Indices Per Sequence Token (List)
    """
    def Join_NER_Sequence_Tokens_To_Original_Form( self, original_sequence, sequence_tokens, sequence_labels ):
        curr_index, new_seq_idx             = 0, 0
        new_sequence, new_labels, label_idx = [], [], []

        # Remove Special Tokens If They Exist
        if "[CLS]" in sequence_tokens:
            sequence_tokens = sequence_tokens[1:]
            sequence_labels = sequence_labels[1:]
        if "[SEP]" in sequence_tokens:
            sequence_tokens = sequence_tokens[0:-1]
            sequence_labels = sequence_labels[0:-1]

        for token, label in zip( sequence_tokens, sequence_labels ):
            token = token.lower()
            # Match The First Pre-Processed Token With Its Matching Token Within The Original Sequence
            #   Store This Token And Its Associated Label + Indices Within Lists
            if new_seq_idx == 0:
                curr_index = original_sequence.lower().index( token, curr_index )
                new_token  = original_sequence[curr_index:curr_index + len( token )]
                new_sequence.append( new_token )
                new_labels.append( label )
                label_idx.append( str( curr_index ) + ":" + str( len( new_token ) ) )
                curr_index += len( token )
                new_seq_idx += 1
            # Now For The More Difficult Portion: Let's Detect The Where The Subsequent Tokens Occur
            #   Within The Original Sequence Text Span. Then Store Them Accordingly With Their Associated
            #   Label And Index Information.
            else:
                curr_index = original_sequence.lower().index( token, curr_index )

                # If The Space Before The Current Token Is Whitespace Within The Original Sequence,
                #   Add The Token To The Current Sequence Along With Its Indices.
                try:
                    # Note: This Will Report/Flag Excepts And Break-Out While Debugging.
                    #       Don't Worry About It. It's Supposed To Do So By Design.
                    whitespace_idx = original_sequence.index( " ", curr_index - 1, curr_index )

                    # If The Above Line Of Code Passes, It Means That The Current Token Is Surrounded By Whitespace.
                    #   Proceed With Adding It To Our Dictionaries.
                    new_token = original_sequence[curr_index:curr_index + len( token )]
                    new_sequence.append( new_token )
                    new_labels.append( label )
                    label_idx.append( str( curr_index ) + ":" + str( curr_index + len( new_token ) ) )
                    new_seq_idx += 1
                except ValueError as e:
                    # Failing The Previous Try Block Means That We've Found A Token Which Is Part Of The Previous Token
                    #   In The Original Sequence Text. BERT's WordPiece Tokenizer Separated Them For One Reason Or Another.
                    #   We Must Join These Back Together Along With Updating The Label And Index Inforamtion Of The Previous Token.
                    #   (Well We Most Likely Want To Ignore This Token Label And Index Information Anyway).
                    #
                    #   Example: Original Seq: "Sandy went to the store, but forgot to bring her wallet."
                    #            Processed   : "Sandy went to the store , but forgot to bring her wallet ."
                    #                                                   ^ - Issue Here Compared To The Original Sequence.
                    new_sequence[new_seq_idx-1] = new_sequence[new_seq_idx-1] + token

                    # Adjust Indices For The New Token Length
                    #  i.e. Not Period At End Of The Sequence Or Comma Separating Sequence Phrases.
                    if token not in [".", ","]:
                        indices = label_idx[new_seq_idx-1].split( ":" )
                        indices[-1] = str( int( indices[-1] ) + len( token ) )
                        label_idx[new_seq_idx-1] = ":".join( indices )


                curr_index += len( token )

        return new_sequence, new_labels, label_idx

    """
        BERT Concept Linking

        Compares BERT Tokenized Sequence Tokens (Sub-Words) To The Original Non-Processed Sequence.
          Returns The Indices For Each Token Within The Sequence Along With The Token Output Labels.
          i.e. Combine Sub-Words Back To Their Originating Tokens By Comparing To The Original Sequence.

        Inputs:
            original_sequence : Non-Processed (Original) Sequence Of Text (List)
            sequence_tokens   : Pre-Processed Tokenized Sequence (List)
            entry_term_mask   : Sub-Word Term Mask (List)
            sequence_labels   : Sub-Word Concept ID Model Predictions (List Of Strings)

        Outputs:
            new_sequence      : New Sequence Of Tokens (List)
            new_term_mask     : Term Mask Associated With Tokens Within 'new_sequence' (List)
            token_idx         : Indices Per Sequence Token (List)
            new_labels        : Sequence Concept ID Labels (List)
    """
    def Combine_CL_Sequence_Sub_Words_To_Original_Form( self, original_sequence, sequence_tokens, entry_term_mask = [], sequence_labels = [] ):
        new_seq_idx = 0
        new_sequence, new_term_mask, token_idx, new_labels = [], [], [], []

        # Check If 'entry_term_mask' Has Been Provided
        if not entry_term_mask:
            self.Print_Log( "BERTBioCreativeDataLoader::Combine_CL_Sequence_Sub_Words_To_Original_Form() - Warning: Entry Term Mask Not Specified / 'entry_term_mask = []'" )
            entry_term_mask = [-1] * len( sequence_tokens )

        # Check If 'sub_word_concepts' Have Been Provided
        #   Note These Are Only Applicable To CLBERTDistributed Model i.e. Per Sub-Word Labels
        if not sequence_labels:
            self.Print_Log( "BERTBioCreativeDataLoader::Combine_CL_Sequence_Sub_Words_To_Original_Form() - Warning: Output Concept Labels Not Specified / 'sequence_labels = []'" )
            sub_word_concepts = [-1] * len( sequence_tokens )
        elif np.asarray( sequence_labels, dtype = np.object ).ndim < 2:
            self.Print_Log( "BERTBioCreativeDataLoader::Combine_CL_Sequence_Sub_Words_To_Original_Form() - Warning: Output Concept Array Is < Dim == 2 / Model Not Build With 'labels_per_sub_word = True'" )
            sub_word_concepts = [-1] * len( sequence_tokens )

        if np.asarray( sequence_labels, dtype = np.object ).ndim == 3:
            sequence_labels = sequence_labels[0]

        # Join Sub-Words Into Tokens
        # NOTE: The First Sub-Word's Entry Term Mask Value Determines
        #       The Desired Token Among The Entire Sequence. This Has
        #       Not Been Tested With Multi-Word Tokens i.e. Chemicals.
        # NOTE: First Sub-Word Label Determines Entry Term Label
        for idx, sub_word in enumerate( sequence_tokens ):
            if re.search( r'^#', sub_word ) and sub_word != "#":
                if len( new_sequence ) > 0:
                    prev_sequence_idx = len( new_sequence ) - 1
                    new_sequence[prev_sequence_idx] += re.sub( r'^#+', "", sub_word )
                else:
                    new_sequence.append( re.sub( r'^#+', "", sub_word ) )
            else:
                new_sequence.append( sub_word )
                new_term_mask.append( entry_term_mask[idx] )
                new_labels.append( sequence_labels[idx] )

        # Replace With New Lists
        sequence_tokens = new_sequence
        entry_term_mask = new_term_mask
        sequence_labels = new_labels

        # Clear Previously Used Variables
        new_sequence, new_term_mask, new_labels, curr_index = [], [], [], 0

        # Remove Special Tokens If They Exist
        if "[PAD]" in sequence_tokens:
            sequence_tokens = [ token for token in sequence_tokens if token != "[PAD]" ]
            entry_term_mask = entry_term_mask[ 0:len( sequence_tokens ) ]
            sequence_labels = sequence_labels[ 0:len( sequence_tokens ) ]
        if "[CLS]" in sequence_tokens:
            sequence_tokens = sequence_tokens[1:]
            entry_term_mask = entry_term_mask[1:]
            sequence_labels = sequence_labels[1:]
        if "[SEP]" in sequence_tokens:
            seq_token_indices = [ idx for idx, token in enumerate( sequence_tokens ) if token == "[SEP]" ]
            sequence_tokens = [ token for idx, token in enumerate( sequence_tokens ) if idx not in seq_token_indices ]
            entry_term_mask = [ value for idx, value in enumerate( entry_term_mask ) if idx not in seq_token_indices ]
            sequence_labels = [ label for idx, label in enumerate( sequence_labels ) if idx not in seq_token_indices ]

        for token, mask_value, token_label in zip( sequence_tokens, entry_term_mask, sequence_labels ):
            token = token.lower()
            # Match The First Pre-Processed Token With Its Matching Token Within The Original Sequence
            #   Store This Token And Its Associated Label + Indices Within Lists
            if new_seq_idx == 0:
                curr_index = original_sequence.lower().index( token, curr_index )
                new_token  = original_sequence[curr_index:curr_index + len( token )]
                new_sequence.append( new_token )
                new_term_mask.append( mask_value )
                new_labels.append( token_label )
                token_idx.append( str( curr_index ) + ":" + str( len( new_token ) ) )
                curr_index += len( token )
                new_seq_idx += 1
            # Now For The More Difficult Portion: Let's Detect The Where The Subsequent Tokens Occur
            #   Within The Original Sequence Text Span. Then Store Them Accordingly With Their Associated
            #   Label And Index Information.
            else:
                curr_index = original_sequence.lower().index( token, curr_index )

                # If The Space Before The Current Token Is Whitespace Within The Original Sequence,
                #   Add The Token To The Current Sequence Along With Its Indices.
                try:
                    # Note: This Will Report/Flag Excepts And Break-Out While Debugging.
                    #       Don't Worry About It. It's Supposed To Do So By Design.
                    whitespace_idx = original_sequence.index( " ", curr_index - 1, curr_index )

                    # If The Above Line Of Code Passes, It Means That The Current Token Is Surrounded By Whitespace.
                    #   Proceed With Adding It To Our Dictionaries.
                    new_token = original_sequence[curr_index:curr_index + len( token )]
                    new_sequence.append( new_token )
                    new_term_mask.append( mask_value )
                    new_labels.append( token_label )
                    token_idx.append( str( curr_index ) + ":" + str( curr_index + len( new_token ) ) )
                    new_seq_idx += 1
                except ValueError as e:
                    # Failing The Previous Try Block Means That We've Found A Token Which Is Part Of The Previous Token
                    #   In The Original Sequence Text. BERT's WordPiece Tokenizer Separated Them For One Reason Or Another.
                    #   We Must Join These Back Together Along With Updating The Label And Index Inforamtion Of The Previous Token.
                    #   (Well We Most Likely Want To Ignore This Token Label And Index Information Anyway).
                    #
                    #   Example: Original Seq: "Sandy went to the store, but forgot to bring her wallet."
                    #            Processed   : "Sandy went to the store , but forgot to bring her wallet ."
                    #                                                   ^ - Issue Here Compared To The Original Sequence.
                    new_sequence[new_seq_idx-1] = new_sequence[new_seq_idx-1] + token

                    # Adjust Indices For The New Token Length
                    #  i.e. Not Period At End Of The Sequence Or Comma Separating Sequence Phrases.
                    if token not in [".", ","]:
                        indices = token_idx[new_seq_idx-1].split( ":" )
                        indices[-1] = str( int( indices[-1] ) + len( token ) )
                        token_idx[new_seq_idx-1] = ":".join( indices )

                curr_index += len( token )

        return new_sequence, new_term_mask, token_idx, new_labels



    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

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
        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - BERT Model Tokenization Handled By Transformers Tokenizer" )

        # ---------------------------------------------------------------------------- #
        #  Generating Unique Token ID Dictionary.                                      #
        #    This Is Not Necessary For The Model, But NERLink Will Throw An Error If   #
        #    The Number Of Unique Terms Is Not Greater Than Zero And Gracefully Exit.  #
        # ---------------------------------------------------------------------------- #

        # Check(s)
        # If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Parameter Settings:" )
        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() -          Lowercase Text: " + str( lowercase ) )

        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Generating Token IDs Using Data" )
        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Processing Data List Elements" )

        # Insert Padding At First Index Of The Token ID Dictionary
        padding_token = self.padding_token.lower() if lowercase else self.padding_token
        if padding_token not in self.token_id_dictionary:
            self.token_id_dictionary[padding_token] = self.number_of_input_tokens
            self.number_of_input_tokens += 1

        # Insert Padding At First Index Of The Token ID Dictionary / Used As 'O' Label In Per-Label CL Classifications
        if padding_token not in self.concept_id_dictionary:
            self.concept_id_dictionary[padding_token] = self.number_of_concept_tokens
            self.number_of_concept_tokens += 1

        # Insert CUI-LESS Token At Second Index Of The Concept ID Dictionary
        cui_less_token = self.cui_less_token.lower() if lowercase else self.cui_less_token
        if cui_less_token not in self.concept_id_dictionary:
            self.concept_id_dictionary[cui_less_token] = self.number_of_concept_tokens
            self.number_of_concept_tokens += 1

        # if padding_token not in self.concept_id_dictionary:
        #     self.concept_id_dictionary[padding_token] = -100

        # Process Sequences In Data List
        for passage in data_list:
            self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Processing Sequence: " + str( passage.Get_Passage() ) )

            # Add Sequence Tokens To Token List
            tokens = passage.Get_Passage().split()

            # Add Concept Linking Entry Terms To Token List
            tokens += passage.Get_Annotations()

            # Check To See If Sequence Tokens Are Already In Dictionary, If Not Add The Tokens
            for token in tokens:
                if lowercase: token = token.lower()
                if token not in self.token_id_dictionary:
                    self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Adding Token: \"" + str( token ) + "\" Value: " + str( self.number_of_input_tokens ) )
                    self.token_id_dictionary[token] = self.number_of_input_tokens
                    self.number_of_input_tokens += 1
                else:
                    self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Adding Token - Warning: \"" + str( token ) + "\" Already In Dictionary" )

            # Build Unique Concept (MESH ID) Dictionary
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
                    # # Remove Padding Token From Concept ID Dictionary
                    # if self.padding_token in self.concept_id_dictionary:
                    #     self.concept_id_dictionary.pop( self.padding_token )
                    #     self.number_of_concept_tokens -= 1

                    self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Adding Concept Token: \"" + str( concept ) + "\" Value: " + str( self.number_of_concept_tokens ) )
                    self.concept_id_dictionary[concept] = self.number_of_concept_tokens
                    self.number_of_concept_tokens += 1

                    # # Ensure Padding Token Is At End Of Concept ID Dictionary
                    # if self.padding_token not in self.concept_id_dictionary:
                    #     self.concept_id_dictionary[self.padding_token] = self.number_of_concept_tokens
                    #     self.number_of_concept_tokens += 1
                else:
                    self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Adding Concept - Warning: \"" + str( concept ) + "\" Already In Dictionary" )

        # Set Number Of Primary Tokens Based On Token ID Dictionary Length
        self.number_of_input_tokens  = len( self.token_id_dictionary )
        self.number_of_output_tokens = len( self.annotation_labels   )

        self.Print_Log( "BERTBioCreativeDataLoader::Generate_Token_IDs() - Complete" )

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
        self.Print_Log( "BERTBioCreativeDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTBioCreativeDataLoader::Load_Vectorized_Model_Data() - Complete" )

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
        self.Print_Log( "BERTBioCreativeDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTBioCreativeDataLoader::Save_Vectorized_Model_Data() - Complete" )

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
        self.Print_Log( "BERTBioCreativeDataLoader::Get_Token_ID() - Fetching ID For Token: \"" + str( token ) + "\"" )
        return self.tokenizer.convert_tokens_to_ids( token )

    """
        Fetches NER Label From ID Value.

        Inputs:
            label_id  : Token (String)

        Outputs:
            ner_label : Token ID Value (Integer)
    """
    def Get_NER_Label_From_ID( self, label_id ):
        self.Print_Log( "BERTBioCreativeDataLoader::Get_NER_Label_From_ID() - Fetching Label From ID: " + str( label_id ) )

        ner_idx_to_labels = { v:k for k, v in self.Get_Annotation_Labels().items() }

        if label_id not in ner_idx_to_labels:
            self.Print_Log( "BERTBioCreativeDataLoader::Get_NER_Label_From_ID() - Label ID Not In NER Labels" )
            return -1

        return ner_idx_to_labels[label_id]


    """
        Fetches Concept Token ID From String.

        Inputs:
            concept    : Token (String)

        Outputs:
            concept_id : Token ID Value (Integer)
    """
    def Get_Concept_ID( self, concept ):
        self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_ID() - Fetching ID For Concept: \"" + str( concept ) + "\"" )

        if self.lowercase_text: concept = concept.lower()

        if concept in self.concept_id_dictionary:
            self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_ID() - Token ID Found: \"" + str( concept ) + "\" => " + str( self.concept_id_dictionary[concept] ) )
            return self.concept_id_dictionary[concept]
        else:
            self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_ID() - Unable To Locate Concept In Dictionary" )

        self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_ID() - Warning: Key Not Found In Dictionary" )

        return -1

    """
        Fetches NER Token String From ID Value.

        Inputs:
            index_value    : Token ID Value (Integer)

        Outputs:
            token_sub_word : Token String (String)
    """
    def Get_Token_From_ID( self, index_value ):
        self.Print_Log( "BERTBioCreativeDataLoader::Get_Token_From_ID() - Searching For ID: " + str( index_value ) )
        return self.tokenizer.convert_ids_to_tokens( index_value )

    """
        Fetches Concept Token String From ID Value.

        Inputs:
            index_value  : Concept ID Value (Integer)

        Outputs:
            key          : Concept TOken String (String)
    """
    def Get_Concept_From_ID( self, index_value ):
        self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_From_ID() - Searching For ID: " + str( index_value ) )

        for key, val in self.concept_id_dictionary.items():
            if val == index_value:
                self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_From_ID() - Found: \"" + str( key ) + "\"" )
                return key

        self.Print_Log( "BERTBioCreativeDataLoader::Get_Concept_From_ID() - Warning: Key Not Found In Dictionary" )

        return None

    def Load_Token_ID_Key_Data( self, file_path ):
        if self.Get_Number_Of_Unique_Tokens() > 0 or self.Get_Number_Of_Unique_Concepts() > 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Load_Token_ID_Key_Data() - Warning: Primary Key Hash Is Not Empty / Saving Existing Data To: \"temp_primary_key_data.txt\"", force_print = True )
            self.Save_Token_ID_Key_Data( "temp_key_data.txt" )

        self.token_id_dictionary, self.concept_id_dictionary = {}, {}

        file_data = self.utils.Read_Data( file_path = file_path )

        if len( file_data ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Load_Token_ID_Key_Data() - Error Loading File Data: \"" + str( file_path ) + "\"" )
            return False

        self.Print_Log( "BERTBioCreativeDataLoader::Load_Token_ID_Data() - Loading Key Data" )

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
            self.Print_Log( "BERTBioCreativeDataLoader::Load_Token_ID_Data() - Key: " + str( key ) + " - Value: " + str( value ) )
            if token_flag:   self.token_id_dictionary[key]   = int( value )
            if concept_flag: self.concept_id_dictionary[key] = int( value )

        self.Print_Log( "BERTBioCreativeDataLoader::Load_Token_ID_Data() - Complete" )

        return True

    def Save_Token_ID_Key_Data( self, file_path ):
        if len( self.token_id_dictionary ) == 0:
            self.Print_Log( "BERTBioCreativeDataLoader::Save_Token_ID_Key_Data() - Warning: Primary Key Data = Empty / No Data To Save" )
            return

        self.Print_Log( "BERTBioCreativeDataLoader::Save_Token_ID_Data() - Saving Key Data" )

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

        self.Print_Log( "BERTBioCreativeDataLoader::Save_Token_ID_Data() - Complete" )

    def Load_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "BERTBioCreativeDataLoader::Load_Vectorized_Model_Data() - Not Implemented / Bypassing Function" )
        pass

    def Save_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "BERTBioCreativeDataLoader::Save_Vectorized_Model_Data() - Not Implemented / Bypassing Function" )
        pass


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
            data_list              : List Of String Instances To Vectorize (Data Chunk Determined By BERTBioCreativeDataLoader::Encode_NER_Model_Data() Function)
            dest_array             : Placeholder For Threaded Function To Store Outputs (Do Not Modify) (List)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays

        Outputs:
            inputs                 : CSR Matrix or Numpy Array
            outputs                : CSR Matrix or Numpy Array

        Note:
            Outputs Are Stored In A List Per Thread Which Is Managed By BERTBioCreativeDataLoader::Encode_NER_Model_Data() Function.

    """
    def Worker_Thread_Function( self, thread_id, data_list, dest_array, use_csr_format = False ):
        self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Vectorizing Data Using Settings" )
        self.Print_Log( "                                                    - Thread ID: " + str( thread_id ) + " - Use CSR Format    : " + str( use_csr_format ) )

        # Vectorized Input/Output Placeholder Lists
        input_sequences    = []
        token_type_ids     = []
        attention_masks    = []
        token_label_ids    = []

        # CSR Matrix Format (Currently Not Used)
        if use_csr_format:
            input_row_index  = 0
            output_row_index = 0

            input_row,  output_row  = [], []
            input_col,  output_col  = [], []
            input_data, output_data = [], []
            output_depth_idx        = []

        # Iterate Through List Of 'Passage' Class Objects
        for passage in data_list:
            self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Text Sequence: " + str( passage.Get_Passage().rstrip() ) )

            # Check
            if not passage.Get_Passage() or len( passage.Get_Passage() ) <= 0:
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Text Contains No Data" )
                continue

            if not passage.Get_Annotations() or len( passage.Get_Annotations() ) == 0:
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Annotation List Contains No Data" )
                continue

            if not passage.Get_Annotation_Labels() or len( passage.Get_Annotation_Labels() ) == 0:
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Error: Passage Annotation Label List Contains No Data" )
                continue

            encoded_sequence, attention_mask, token_type_id, encoded_label_sequence = self.Encode_NER_Instance( passage.Get_Passage().rstrip(), passage.Get_Annotations(), passage.Get_Annotation_Labels(),
                                                                                                                passage.Get_Annotation_Indices(), composite_mention_list = passage.Get_Composite_Mention_List(),
                                                                                                                individual_mention_list = passage.Get_Individual_Mention_List() )

            # Check(s)
            if len( encoded_sequence ) == 0 or len( encoded_label_sequence ) == 0:
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Error Occurred While Encoding Text Sequence", force_print = True )
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() -           Text Sequence      : '" + str( passage.Get_Passage().rstrip()  ) + "'", force_print = True )
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() -           Annotations        : '" + str( passage.Get_Annotations()       ) + "'", force_print = True )
                self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() -           Annotations Labels : '" + str( passage.Get_Annotation_Labels() ) + "'", force_print = True )
                continue

            ############################
            # Input & Output Sequences #
            ############################
            input_sequences.append( encoded_sequence )
            attention_masks.append( attention_mask )
            token_type_ids.append( token_type_id )
            token_label_ids.append( encoded_label_sequence )

        self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Input Data :\n" + str( input_sequences ) )
        self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data:\n" + str( token_label_ids ) )

        input_sequences = np.asarray( input_sequences )
        token_label_ids = np.asarray( token_label_ids )

        # Check(s)
        if len( input_sequences ) == 0:
            dest_array[thread_id] = None
            return

        # Assign Thread Vectorized Data To Temporary DataLoader Placeholder Array
        dest_array[thread_id] = [input_sequences, attention_masks, token_type_ids, token_label_ids]

        self.Print_Log( "BERTBioCreativeDataLoader::Worker_Thread_Function() - Complete" )



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NERLink.DataLoader import BERTBioCreativeDataLoader\n" )
    print( "     data_loader = BERTBioCreativeDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

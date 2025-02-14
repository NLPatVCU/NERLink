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

import re
import numpy as np
from sparse import COO

import tensorflow as tf

#tf.logging.set_verbosity( tf.logging.ERROR )                       # Tensorflow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # Tensorflow v1.x

# Tensorflow Version Check - BERT DataLoader Only Supports Tensorflow Versions >= 2.x
if re.search( r"^2.\d+", tf.__version__ ):
    import transformers
    transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements

# Custom Modules
from NERLink.DataLoader.BioCreative import BERTBioCreativeDataLoader


############################################################################################
#                                                                                          #
#   BERT Data Loader Model Class                                                           #
#                                                                                          #
############################################################################################

class BERTBioCreativeMultiTaskDataLoader( BERTBioCreativeDataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  bert_model = "bert-base-cased", skip_individual_mentions = False, skip_composite_mentions = False, lowercase = False, ignore_label_type_list = [] ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, skip_out_of_vocabulary_words = skip_out_of_vocabulary_words,
                          debug_log_file_handle = debug_log_file_handle, bert_model = bert_model, skip_individual_mentions = skip_individual_mentions,
                          skip_composite_mentions = skip_composite_mentions, shuffle = shuffle, lowercase = lowercase, ignore_label_type_list = ignore_label_type_list )
        self.version = 0.04

    """
        Encodes NER-Concept Linking Data - Used For Training, Validation Or Evaluation Data

        Inputs:
            data_list               : List Of Passage Objects
            use_csr_format          : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            keep_in_memory          : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            is_validation_data      : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data      : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
            pad_input               : Adds Padding To Input Sequence ie. [PAD] Tokens After Actual Sequence Until Max Sequence Length (Bool)
            pad_output              : Produces An Entire Vector For A Given Instance. ie. [0, 1, 0, 0] vs [2]
                                      Categorical Crossentropy vs. Sparse Categorical Crossentropy
            concept_delimiter       : Concept ID Delimiter Used To Separate Concept IDs Given A Single Instance (String/None)
                                      Used For One-To-Many Relationships
            mask_term_sequence      : True  -> Entry Term Mask = Entire Sub-Word Sequence Containing Entry Term
                                      False -> Encode Just Entry Term Sub-Word Tokens
            separate_sentences      : Separates Sentences With [SEP] Token Using Sentence Delimiters (Bool)
            term_sequence_only      : Disregards All Sequences Surrounding The Sequence Of Interest, Only Encoding The Desired Sequence (Bool)
            restrict_context        : Restricts Or Reduces The Sequence Context Surrounding The Entry Term Used To Generate Its Embedding (Bool)
            label_per_sub_word      : Produces An Output Label For Each Sub-Word Token In The Sequence (Bool)
            use_cui_less_labels     : Sets All Non-CUI-Less Labels To CUI-Less, Not Padding Label (i.e. 1, not 0) (Bool)
            split_by_max_seq_length : Splits Single Instance Exceeding Max Sequence Length Into Multiple Instances (Avoids Truncation) (Bool)
            ignore_output_errors    : Continues Encoding Output Labels Data When Error Are Found (i.e. Concept ID Not In Dictionary) (Bool)

        Outputs:
            encoded_inputs          : Tuple Of Encoded Concept Model Inputs
                                            1) Encoded Token IDs
                                            2) Attention Masks
                                            3) Token Type IDs
                                            4) Entry Term Masks
            encoded_outputs         : Tuple Of Encoded Outputs
                                            1) Encoded NER Labels (CSR, COO Matrix or Numpy Array)
                                            2) Encoded Concept Labels (CSR, COO Matrix or Numpy Array)
    """
    def Encode_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = True, keep_in_memory = True,
                           is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                           mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = True,
                           use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None

        cl_pad_output = False

        if not pad_output:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Warning: CL Output Requires 'pad_output = True' / Forcing 'cl_pad_output = True'", force_print = True )
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -          NOTE: This Does Not Pad NER Output While Padding CL Output",              force_print = True )
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -                i.e. Sparse Categorical Crossentropy v Binary Crossentropy Labels", force_print = True )
            cl_pad_output = True
        elif label_per_sub_word and pad_output:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Warning: Large Memory Consumption When 'label_per_sub_word = True' and 'pad_output = True' / Setting 'pad_output = False'" )
            pad_output    = False
            cl_pad_output = True

        # Clear Previous Concept Instance Data Index List
        self.concept_instance_data_idx.clear()

        # Enforce BERT Max Sequence Length Limitation
        if self.max_sequence_length > self.max_sequence_limit:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Warning: Max Sequence Length > " + str( self.max_sequence_limit ) + " / Enforcing BERT Max Sequence Length == "  + str( self.max_sequence_limit ) )
            self.max_sequence_length = self.max_sequence_limit

        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Encoding Concept Instances" )

        encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_term_masks, encoded_ner_outputs, encoded_concept_outputs = [], [], [], [], [], []
        ner_output_row, ner_output_col, ner_output_depth, ner_output_data = [], [], [], []
        cl_output_row, cl_output_col, cl_output_depth, cl_output_data     = [], [], [], []
        output_instance_index, sequence_length, annotation_id_to_idx      = 0, 0, {}
        concept_cui_less_id                                               = self.Get_Concept_ID( self.Get_CUI_LESS_Token() )
        concept_padding_id                                                = self.Get_Concept_ID( self.Get_Padding_Token() )
        ner_outside_token_id                                              = self.annotation_labels["O"]
        default_concept_cui_less_id                                       = concept_cui_less_id if use_cui_less_labels else concept_padding_id

        for passage_idx, passage in enumerate( data_list ):
            passage_sequence        = passage.Get_Passage().rstrip()
            annotations             = passage.Get_Annotations()
            annotation_indices      = passage.Get_Annotation_Indices()
            annotation_labels       = passage.Get_Annotation_Labels()
            annotation_concept_ids  = passage.Get_Annotation_Concept_IDs()
            composite_mention_list  = passage.Get_Composite_Mention_List()
            individual_mention_list = passage.Get_Individual_Mention_List()

            # Check(s)
            if passage_sequence == "" or len( passage_sequence ) == 0:
                self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error: Text Sequence Is Empty String" )
                continue

            if len( annotations ) == 0 or len( annotation_labels ) == 0 or len( annotation_concept_ids ) == 0:
                self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error: Passage Contains No NER or CL Annotations/Labels" )
                self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -        Passage: " + str( passage.Get_Passage() ) )
                continue

            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Encoding Inputs" )
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -                   Text Sequence         : " + str( passage_sequence      ) )
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -                   Annotations           : " + str( annotations           ) )
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() -                   Annotation Indices    : " + str( annotation_indices    ) )

            # Determine Which Entities Have Annotations
            curr_annotation_index        = 0
            curr_offset                  = 0
            text_sequence_character_mask = [ "_" if passage_sequence[i] != " " else " " for i in range( len( passage_sequence ) ) ]

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
                    extracted_token   = passage_sequence[annotation_offset:annotation_end]

                    if number_of_indices == 1 and annotation != extracted_token:
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() - Error: Extracted Token != True Token", force_print = True )
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() -        True Token: " + str( annotation ), force_print = True )
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() -        Extracted Token: " + str( extracted_token ), force_print = True )
                        continue
                    elif number_of_indices > 1 and extracted_token not in annotation:
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() - Error: Extracted Token Not In True Token", force_print = True )
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() -        True Token: " + str( annotation ), force_print = True )
                        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_CL_Model_Data() -        Extracted Token: " + str( extracted_token ), force_print = True )
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

            # Encoding Error / No Annotations Matched In Mask
            if len( [ True for token in text_sequence_annotation_indices if re.search( r'\d+', token ) is not None ] ) == 0: continue

            ###############################################
            # Encode Text Sequence And Set Attention Mask #
            ###############################################

            prev_annotation_id, encoding_error = 0, False
            subword_tokens, processed_annotation_ids, term_mask, subword_ner_label_ids, subword_concept_label_ids = [], [], [], [], []

            # Tokenize Data Into Sub-Words And Assign Label IDs Per Sub-Word
            for ( index, token ), token_annotation_id in zip( enumerate( passage_sequence.split() ), text_sequence_annotation_indices ):
                # Check
                if encoding_error: break

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
                    term_mask.extend( [0] * len( token_subwords ) )
                    subword_ner_label_ids.extend( [self.annotation_labels["O"]] * len( token_subwords ) )
                    subword_concept_label_ids.extend( [[default_concept_cui_less_id]] * len( token_subwords ) )

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
                        subword_ner_label = annotation_labels[annotation_id_to_idx[annotation_id]]

                        # Get Annotation Concept IDs - Singular And Complex Supported
                        concept_ids = []
                        for concept in annotation_concept_ids[annotation_id_to_idx[annotation_id]].split( concept_delimiter ):
                            concept_ids.append( self.Get_Concept_ID( concept ) )

                        # Check
                        if ignore_output_errors and -1 in concept_ids:
                            desired_concepts = annotation_concept_ids[annotation_id_to_idx[annotation_id]].split( concept_delimiter )
                            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Warning: Concept ID Not Found While Encoding Output" )
                            self.Print_Log( "                                                        - Desired Concept(s)    : " + str( desired_concepts ) )
                            self.Print_Log( "                                                        - Obtained Concept iD(s): " + str( concept_ids      ) )
                            concept_ids = [default_concept_cui_less_id]
                        elif not ignore_output_errors and -1 in concept_ids:
                            desired_concepts = annotation_concept_ids[annotation_id_to_idx[annotation_id]].split( concept_delimiter )
                            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error Encoding Concept ID Output / Concept ID Not Found In Desired Concept(s)" )
                            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Desired Concept(s)    : " + str( desired_concepts ) )
                            self.Print_Log( "                                                        - Obtained Concept iD(s): " + str( concept_ids      ) )
                            encoding_error = True
                            break

                        temp_word, start_tagged = "", False if annotation_id not in processed_annotation_ids else True
                        current_idx = len( subword_ner_label_ids ) - ( len( token_subwords ) )

                        # Tag The Specific Start & End Indices With The Appropriate NER Label
                        for token_idx, sub_word in enumerate( token_subwords ):
                            if sub_word != "#": sub_word = re.sub( r'^#+', "", sub_word )
                            temp_word        += sub_word
                            temp_word_length = len( temp_word ) - 1 - annotation_start

                            # If Annotation Start Index Equals Or Within Current Sub-Word, Tag Sub-Word Token As Beginning Label
                            if not start_tagged and len( temp_word ) - 1 >= annotation_start:
                                term_mask[current_idx + token_idx]                 = annotation_id + 1
                                subword_ner_label_ids[current_idx + token_idx]     = int( self.annotation_labels["B-" + str( subword_ner_label )] )
                                subword_concept_label_ids[current_idx + token_idx] = concept_ids
                                start_tagged = True
                            # Continue Tagging Next Sub-Word Tokens As 'Intermediate' Label Of Preceeding Label
                            elif start_tagged and len( temp_word ) - 1 >= annotation_start and temp_word_length < annotation_length:
                                term_mask[current_idx + token_idx]                 = annotation_id + 1
                                subword_ner_label_ids[current_idx + token_idx]     = int( self.annotation_labels["I-" + str( subword_ner_label )] )
                                subword_concept_label_ids[current_idx + token_idx] = concept_ids
                            # We've Exceeded The Actual Annotation Label Span, Discontinue Modifying NER Label As 'Beginning' Or 'Intermediate'
                            elif temp_word_length > annotation_length:
                                break

                        # Add Current Annotation ID To Processed Annotation IDs
                        if annotation_id not in processed_annotation_ids: processed_annotation_ids.append( annotation_id )
                else:
                    term_mask.extend( [0] * len( token_subwords ) )
                    subword_ner_label_ids.extend( [self.annotation_labels["O"]] * len( token_subwords ) )
                    subword_concept_label_ids.extend( [[concept_cui_less_id]] * len( token_subwords ) )

            # Check For Error Occurring During Output Encoding And Skip Passage Instance If Detected
            if encoding_error:
                self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error Encoding Passage: " + str( passage_sequence ), force_print = True )
                continue

            # Truncate By Max Sequence Length And Make Room To Add The [CLS] and [SEP] Special Tokens
            if not split_by_max_seq_length and len( subword_tokens ) > self.max_sequence_length - self.special_tokens_count:
                term_mask                 = term_mask[: ( self.max_sequence_length - self.special_tokens_count ) ]
                subword_tokens            = subword_tokens[: ( self.max_sequence_length - self.special_tokens_count ) ]
                subword_ner_label_ids     = subword_ner_label_ids[: ( self.max_sequence_length - self.special_tokens_count ) ]
                subword_concept_label_ids = subword_concept_label_ids[: ( self.max_sequence_length - self.special_tokens_count ) ]

            # Encode Sub-Word Tokens Into BERT Inputs: Token IDs, Attention Masks and Token Type IDs
            max_length = self.max_sequence_length if not split_by_max_seq_length else len( subword_tokens ) + self.special_tokens_count
            inputs     = self.tokenizer.encode_plus( subword_tokens, add_special_tokens = True, max_length = max_length )

            token_ids, attention_mask, token_type_ids = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
            term_mask          = [0] + term_mask + [0]
            encoded_ner_labels = [self.label_sequence_padding] + subword_ner_label_ids + [self.label_sequence_padding]
            encoded_cl_labels  = [[default_concept_cui_less_id]] + subword_concept_label_ids + [[default_concept_cui_less_id]]

            # Pad Sequences
            if ( pad_input or pad_output or cl_pad_output ) and self.max_sequence_length - len( token_ids ) > 0:
                padding_length = self.max_sequence_length - len( token_ids )
                token_ids.extend( [self.sub_word_pad_token_id] * padding_length )
                attention_mask.extend( [self.sub_word_pad_token_id] * padding_length )
                token_type_ids.extend( [self.sub_word_pad_token_id] * padding_length )
                term_mask.extend( [0] * padding_length )
                encoded_ner_labels.extend( [ner_outside_token_id] * padding_length )
                encoded_cl_labels.extend( [[concept_padding_id]] * padding_length )

            # Get Length Of Tokenized Sequence
            sequence_length = len( token_ids )

            # Split Sequences Over Max Sequence Length, Supported By The BERT Tokenizer, Into Individual Instances
            #   This Attempts To Avoid Sub-Word Truncation For Instances Exceeding The Maximum Sequence Length Supported By The Model
            if split_by_max_seq_length and sequence_length > self.Get_Max_Sequence_Length():
                split_instance_length          = self.Get_Max_Sequence_Length() - self.special_tokens_count
                split_start_idx, split_end_idx = 0, split_instance_length # Make Room For [CLS] and [SEP] Tokens
                sequence_end_idx               = token_ids.index( self.Get_Sub_Word_SEP_Token_ID() )
                skip_remaining_splits          = False

                # Perform Instance Splits
                while split_start_idx < sequence_length:
                    # Skip Remaining Splits If We Encounter The Sequence End i.e. [SEP] Token
                    if split_start_idx >= sequence_end_idx: break

                    # Check For CL Output Instance At Max Sequence Length / A CL Mapping Exists At The Last Element Of The Concept Output Array
                    if encoded_cl_labels[split_end_idx] not in [[concept_padding_id], [concept_cui_less_id]]:
                        # Find Nearest CUI-Less Or Padding To The Left Of The Last Token
                        #   This Will Be The End Of The First Split And Beginning Of The Next Split
                        temp_split_end_idx = split_end_idx

                        # Keep Decreasing The Temporary End Index Until We Find A Padding Or CUI-Less Label
                        while temp_split_end_idx > split_start_idx and encoded_cl_labels[temp_split_end_idx] not in [[concept_padding_id], [concept_cui_less_id]]:
                            temp_split_end_idx -= 1

                        # We Found A New Split Without The CL Instance At The End
                        if temp_split_end_idx > split_start_idx:
                            split_end_idx = temp_split_end_idx + 1

                    # Split Instance Into Multiple Instances (Instance Length Exceeds The Tokenizer Max Length)
                    split_token_ids          = token_ids[split_start_idx:split_end_idx]
                    split_attention_mask     = attention_mask[split_start_idx:split_end_idx]
                    split_token_type_ids     = token_type_ids[split_start_idx:split_end_idx]
                    split_term_mask          = term_mask[split_start_idx:split_end_idx]
                    split_encoded_ner_output = encoded_ner_labels[split_start_idx:split_end_idx]
                    split_encoded_cl_output  = encoded_cl_labels[split_start_idx:split_end_idx]

                    # End Of Sequence Detected / Skip Remaining Splits
                    if self.Get_Sub_Word_SEP_Token_ID() in split_token_ids:
                        skip_remaining_splits = True

                    # Skip Token ID Split Without Actual Tokens Or No Non-CUI-Less/Padding CL Instance Found In Instance Split / Skipping Split Instance
                    if skip_remaining_splits or all( val == 0 for val in split_term_mask ):
                        split_start_idx = split_end_idx
                        split_end_idx   = split_start_idx + split_instance_length if split_start_idx + split_instance_length < sequence_length else sequence_length - 1
                        continue

                    # Check For Special Token i.e. [CLS] (101) Token
                    #   If Not Add [CLS] Token To Inputs ID List And Pad Remaining Inputs/Output Lists
                    if split_token_ids[0]  != self.sub_word_cls_token_id:
                        split_token_ids.insert( 0, self.sub_word_cls_token_id )
                        split_attention_mask.insert( 0, 1 )
                        split_token_type_ids.insert( 0, 0 )
                        split_term_mask.insert( 0, 0 )
                        split_encoded_ner_output.insert( 0, ner_outside_token_id )
                        split_encoded_cl_output.insert( 0, [default_concept_cui_less_id] )

                    # Check For Special Token i.e. [SEP] (102) Token
                    #   If Not Add [SEP] Token To Inputs ID List And Pad Remaining Inputs/Output Lists
                    if split_token_ids[-1] != self.sub_word_sep_token_id:
                        split_token_ids.append( self.sub_word_sep_token_id )
                        split_attention_mask.append( 1 )
                        split_token_type_ids.append( 0 )
                        split_term_mask.append( 0 )
                        split_encoded_ner_output.append( ner_outside_token_id )
                        split_encoded_cl_output.append( [default_concept_cui_less_id] )

                    # Add Padding If Required i.e. Pad To Max Sequence Length
                    if self.Get_Max_Sequence_Length() - len( split_token_ids ) > 0:
                        padding_length = self.Get_Max_Sequence_Length() - len( split_token_ids )
                        split_token_ids.extend( [self.sub_word_pad_token_id] * padding_length )
                        split_attention_mask.extend( [self.sub_word_pad_token_id] * padding_length )
                        split_token_type_ids.extend( [self.sub_word_pad_token_id] * padding_length )
                        split_term_mask.extend( [self.sub_word_pad_token_id] * padding_length )
                        split_encoded_ner_output.extend( [ner_outside_token_id] * padding_length )
                        split_encoded_cl_output.append( [concept_padding_id] * padding_length )

                    # Store Encoded Instance Elements In Their Appropriate Lists
                    encoded_token_ids.append( split_token_ids )
                    encoded_attention_masks.append( split_attention_mask )
                    encoded_token_type_ids.append( split_token_type_ids )
                    encoded_term_masks.append( split_term_mask )

                    # Concept Output Is A Vector/Array Of 'N' Classes With Our Desired Instance Class As '1'
                    #   i.e. [0, 0, 1, 0, 0]
                    #   Append An NER & CL Instance Labels
                    if use_csr_format:
                        # NER Instances Will Generally Be In 'Sparse_Categorical_Crossentropy' Format
                        #   So The COO Matrix Will Always Be 2D
                        for i, value in enumerate( split_encoded_ner_output ):
                            if value == 0: continue

                            ner_output_row.append( output_instance_index )
                            ner_output_col.append( i )

                            # Pad NER Output
                            if pad_output:
                                ner_output_depth.append( value )
                                ner_output_data.append( 1 )
                            # Non-Padded NER Output
                            else:
                                ner_output_data.append( value )

                        # CL Output Matrix Can Be 2D or 3D, Depending On Loss Function
                        #   Sparse/Categorical Crossentropy Or Binary Crossentropy
                        for i, values in enumerate( split_encoded_cl_output ):
                            if not label_per_sub_word and value == 0: continue

                            # Pad CL Output
                            if label_per_sub_word and ( pad_output or cl_pad_output ):
                                for value in values:
                                    cl_output_row.append( output_instance_index )
                                    cl_output_col.append( i )
                                    cl_output_depth.append( value )
                                    cl_output_data.append( 1 )
                            # Non-Padded CL Output
                            else:
                                for value in values:
                                    cl_output_row.append( output_instance_index )
                                    cl_output_col.append( i )
                                    cl_output_data.append( value )

                        output_instance_index += 1
                    else:
                        encoded_ner_outputs.append( split_encoded_ner_output )
                        encoded_concept_outputs.append( split_encoded_cl_output )

                    # Keep Track Of Which Passage The Instance Came From
                    self.concept_instance_data_idx.append( passage_idx )

                    # Update Indices
                    split_start_idx = split_end_idx
                    split_end_idx   = split_start_idx + split_instance_length if split_start_idx + split_instance_length < sequence_length else sequence_length - 1
            # No Splits Required
            else:
                # Store Encoded Instance Elements In Their Appropriate Lists
                encoded_token_ids.append( token_ids )
                encoded_attention_masks.append( attention_mask )
                encoded_token_type_ids.append( token_type_ids )
                encoded_term_masks.append( term_mask )

                # Concept Output Is A Vector/Array Of 'N' Classes With Our Desired Instance Class As '1'
                #   i.e. [0, 0, 1, 0, 0]
                #   Append An NER & CL Instance Labels
                if use_csr_format:
                    # NER Instances Will Generally Be In 'Sparse_Categorical_Crossentropy' Format
                    #   So The COO Matrix Will Always Be 2D
                    for i, value in enumerate( encoded_ner_labels ):
                        if value == 0: continue

                        ner_output_row.append( output_instance_index )
                        ner_output_col.append( i )

                        # Pad NER Output
                        if pad_output:
                            ner_output_depth.append( value )
                            ner_output_data.append( 1 )
                        # Non-Padded NER Output
                        else:
                            ner_output_data.append( value )

                    # CL Output Matrix Can Be 2D or 3D, Depending On Loss Function
                    #   Sparse/Categorical Crossentropy Or Binary Crossentropy
                    for i, values in enumerate( encoded_cl_labels ):
                        if not label_per_sub_word and value == 0: continue

                        # Pad CL Output
                        if label_per_sub_word and ( pad_output or cl_pad_output ):
                            for value in values:
                                cl_output_row.append( output_instance_index )
                                cl_output_col.append( i )
                                cl_output_depth.append( value )
                                cl_output_data.append( 1 )
                        # Non-Padded CL Output
                        else:
                            for value in values:
                                cl_output_row.append( output_instance_index )
                                cl_output_col.append( i )
                                cl_output_data.append( value )

                    output_instance_index += 1
                else:
                    encoded_ner_outputs.append( encoded_ner_labels )
                    encoded_concept_outputs.append( encoded_cl_labels )

                # Keep Track Of Which Passage The Instance Came From
                self.concept_instance_data_idx.append( passage_idx )

            # Clear Annotation ID To Index Mapping
            annotation_id_to_idx.clear()

        # Convert Data To Numpy Arrays
        encoded_token_ids        = np.asarray( encoded_token_ids,       dtype = np.int32 )
        encoded_attention_masks  = np.asarray( encoded_attention_masks, dtype = np.int32 )
        encoded_token_type_ids   = np.asarray( encoded_token_type_ids,  dtype = np.int32 )
        encoded_term_masks       = np.asarray( encoded_term_masks,      dtype = np.int32 )

        # Convert Into COO Matrix
        if use_csr_format:
            number_of_output_rows   = sequence_length
            ner_output_data         = np.asarray( ner_output_data, dtype = np.int32 )
            number_of_labels        = len( self.Get_Annotation_Labels() )

            if pad_output:
                encoded_ner_outputs = COO( [ ner_output_row, ner_output_col, ner_output_depth ], ner_output_data, shape = ( output_instance_index, number_of_output_rows, number_of_labels ), fill_value = 0 )
            else:
                encoded_ner_outputs = COO( [ ner_output_row, ner_output_col ], ner_output_data, shape = ( output_instance_index, number_of_output_rows ), fill_value = 0 )

            cl_output_data          = np.asarray( cl_output_data, dtype = np.int32 )
            number_of_labels        = self.Get_Number_Of_Unique_Concepts()

            if label_per_sub_word and ( pad_output or cl_pad_output ):
                encoded_concept_outputs = COO( [ cl_output_row, cl_output_col, cl_output_depth ], cl_output_data, shape = ( output_instance_index, number_of_output_rows, number_of_labels ), fill_value = 0 )
            else:
                encoded_concept_outputs = COO( [ cl_output_row, cl_output_col ], cl_output_data, shape = ( output_instance_index, number_of_output_rows ), fill_value = 0 )
        else:
            encoded_ner_outputs     = np.asarray( encoded_ner_outputs,     dtype = np.int32 )
            encoded_concept_outputs = np.asarray( encoded_concept_outputs, dtype = np.int32 )

        # Check(s)
        number_of_input_instances  = encoded_token_ids.shape[0]
        number_of_output_instances = encoded_concept_outputs.shape[0] if isinstance( encoded_concept_outputs, COO ) else len( encoded_concept_outputs )

        if number_of_input_instances == 0:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error Occurred While Encoding Concept Inputs", force_print = True )
            return None, None
        elif number_of_output_instances == 0:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error Occurred While Encoding Concept Outputs", force_print = True )
            return None, None
        elif number_of_input_instances != number_of_output_instances:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Error: Number Of Input And Output Instances Not Equal", force_print = True )
            return None, None

        if keep_in_memory:
            self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Storing Encoded Data In Memory" )

            if is_validation_data:
                self.ner_val_inputs       = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids )
                self.ner_val_outputs      = ( encoded_ner_outputs, encoded_concept_outputs )
                self.concept_val_inputs   = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_term_masks )
                self.concept_val_outputs  = ( encoded_ner_outputs, encoded_concept_outputs )
            elif is_evaluation_data:
                self.ner_eval_inputs      = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids )
                self.ner_eval_outputs     = ( encoded_ner_outputs, encoded_concept_outputs )
                self.concept_eval_inputs  = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_term_masks )
                self.concept_eval_outputs = ( encoded_ner_outputs, encoded_concept_outputs )
            else:
                self.ner_inputs           = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids )
                self.ner_outputs          = ( encoded_ner_outputs, encoded_concept_outputs )
                self.concept_inputs       = ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_term_masks )
                self.concept_outputs      = ( encoded_ner_outputs, encoded_concept_outputs )

        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Encode_Model_Data() - Complete" )

        return ( encoded_token_ids, encoded_attention_masks, encoded_token_type_ids, encoded_term_masks ), ( encoded_ner_outputs, encoded_concept_outputs )

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)
            file_name : File Name (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path, file_name = "" ):
        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Load_Vectorized_Model_Data() - Complete" )

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
        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # ToDo
        raise NotImplementedError

        self.Print_Log( "BERTBioCreativeMultiTaskDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False


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
    print( "     from NERLink.DataLoader import BERTBioCreativeMultiTaskDataLoader\n" )
    print( "     data_loader = BERTBioCreativeMultiTaskDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

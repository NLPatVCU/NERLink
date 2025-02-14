#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    10/08/2020                                                                   #
#    Revised: 11/19/2022                                                                   #
#                                                                                          #
#    Base Data Loader Class For The NERLink Package.                                       #
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

# Custom Modules
from NERLink.Misc import Utils


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class DataLoader( object ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False,
                  debug_log_file_handle = None, lowercase = False, skip_individual_mentions = False, skip_composite_mentions = False,
                  ignore_label_type_list = [] ):
        self.version                      = 0.12
        self.debug_log                    = print_debug_log                 # Options: True, False
        self.write_log                    = write_log_to_file               # Options: True, False
        self.debug_log_file_handle        = debug_log_file_handle           # Debug Log File Handle
        self.debug_log_file_name          = "DataLoader_Log.txt"            # File Name (String)
        self.token_id_dictionary          = {}                              # Token ID Dictionary: Maps Tokens To Token IDs
        self.concept_frequency_dictionary = {}                              # Keeps Track Of Concept Frequencies
        self.concept_id_dictionary        = {}                              # Concept ID Dictionary: Maps Concepts To Concept IDs
        self.concept_instance_data_idx    = []                              # Each Element Corresponds An Element In 'self.concept_input' Which Denote Where In 'self.data_list' The Input Instance Originated From. (List)
        self.annotation_labels            = {}                              # NER Entity Label Dictionary: Maps NER Entity Labels To Entity Label IDs
        self.oov_term_dict                = {}                              # Keeps Track Of Out Of Vocabulary Terms And Their Respective Frequencies
        self.ignore_label_type_list       = ignore_label_type_list          # List Of Labels To Skip/Omot When Reading Data Prior To Encoding (NER + CL)
        self.skip_out_of_vocabulary_words = skip_out_of_vocabulary_words    # Options: True, False
        self.skip_individual_mentions     = skip_individual_mentions        # Skips CL 'IndividualMention' Annotation/Entities (Bool)
        self.skip_composite_mentions      = skip_composite_mentions         # Skips CL 'CompositeMention' Annotation/Entities (Bool)
        self.number_of_input_tokens       = 0
        self.number_of_concept_tokens     = 0
        self.max_sequence_length          = 0
        self.label_sequence_padding       = 0
        self.pad_label_token_id           = 0
        self.pad_token_segment_id         = 0
        self.sequence_a_segment_id        = 0
        self.data_list                    = []
        self.ner_inputs                   = None
        self.ner_outputs                  = None
        self.concept_inputs               = None
        self.concept_outputs              = None
        self.ner_val_inputs               = None
        self.ner_val_outputs              = None
        self.concept_val_inputs           = None
        self.concept_val_outputs          = None
        self.ner_eval_inputs              = None
        self.ner_eval_outputs             = None
        self.concept_eval_inputs          = None
        self.concept_eval_outputs         = None
        self.embeddings_a                 = []
        self.embeddings_b                 = []
        self.embeddings_a_loaded          = False
        self.embeddings_b_loaded          = False
        self.simulate_embeddings_a_loaded = False
        self.simulate_embeddings_b_loaded = False
        self.read_file_handle             = None
        self.generated_embedding_ids      = False
        self.separated_by_input_type      = False
        self.max_sequence_length_set      = False
        self.lowercase_text               = lowercase
        self.padding_token                = "<*>PADDING<*>"
        self.cui_less_token               = "<*>CUI-LESS<*>"
        self.sub_word_cls_token           = "[CLS]"
        self.sub_word_sep_token           = "[SEP]"
        self.sub_word_pad_token           = "[PAD]"
        self.sub_word_cls_token_id        = 0
        self.sub_word_sep_token_id        = 0
        self.sub_word_pad_token_id        = 0
        self.data_file_path               = ""
        self.cl_accepted_labels           = [ "chemical", "disease", 'diseaseclass', 'specificdisease', 'compositemention', 'modifier' ]
        self.utils                        = Utils()

        # Create Log File Handle
        if self.write_log and self.debug_log_file_handle is None:
            self.debug_log_file_handle = open( self.debug_log_file_name, "w" )

    """
        Remove Variables From Memory
    """
    def __del__( self ):
        self.Clear_Data()
        self.Close_Read_File_Handle()
        del self.utils
        if self.write_log and self.debug_log_file_handle is not None: self.debug_log_file_handle.close()

    """
        Performs Checks Against The Specified Data File/Data List To Ensure File Integrity Before Further Processing
            (Only Checks First 10 Lines In Data File)
    """
    def Check_Data_File_Format( self, file_path = "", data_list = [], is_crichton_format = False, str_delimiter = '\t' ):
        self.Print_Log( "DataLoader::Check_Data_File_Format() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized NER Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list              : List Of Variables
    """
    def Encode_NER_Model_Data( self, data_list = [] ):
        self.Print_Log( "DataLoader::Encode_NER_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Concept Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list              : List Of Variables
    """
    def Encode_CL_Model_Data( self, data_list = [] ):
        self.Print_Log( "DataLoader::Encode_CL_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Tokenizes Data Sequences Into List Of Tokens With Or Without Padding
            Used For ELMo Implementation

        Inputs:
            data_list              : List Of Variables
    """
    def Tokenize_Model_Data( self, data_list = [], use_padding = True ):
        self.Print_Log( "DataLoader::Tokenize_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Model Data - Single Input Instance And Output Instance
    """
    def Encode_NER_Instance( self ):
        self.Print_Log( "DataLoader::Encode_NER_Input_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Model Data - Single Input Instance And Output Instance
    """
    def Encode_CL_Input_Instance( self ):
        self.Print_Log( "DataLoader::Encode_CL_Input_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Model Data - Single Input Instance And Output Instance
    """
    def Encode_CL_Output_Instance( self ):
        self.Print_Log( "DataLoader::Encode_CL_Output_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Model Data - Single Input Instance And Output Instance
    """
    def Encode_CL_Instance( self ):
        self.Print_Log( "DataLoader::Encode_CL_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Sequence Of Token IDs to Sequence Of Token Strings
    """
    def Decode_NER_Input_Instance( self ):
        self.Print_Log( "DataLoader::Decode_NER_Input_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings
    """
    def Decode_NER_Output_Instance( self ):
        self.Print_Log( "DataLoader::Decode_NER_Output_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Input & Output Sequence Of NER Token IDs And Label IDs To Sequence Of NER Token & Label Strings
    """
    def Decode_NER_Instance( self ):
        self.Print_Log( "DataLoader::Decode_NER_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Input Sequence Instance Of IDs To Entry Term String(s)
    """
    def Decode_CL_Input_Instance( self ):
        self.Print_Log( "DataLoader::Decode_CL_Input_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Output Instance Of Labels For Concept Linking To List Of Concept ID Strings
    """
    def Decode_CL_Output_Instance( self ):
        self.Print_Log( "DataLoader::Decode_CL_Output_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self ):
        self.Print_Log( "DataLoader::Decode_CL_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Decodes Input & Output Sequence Of NER Token IDs And Label IDs To Sequence Of NER Token & Label Strings. Also Writes This Decoded
          Data Back To A Formatted File.
    """
    def Write_Formatted_File( self, write_file_path = "output_file", data_list = [], encoded_ner_inputs = None,
                              encoded_ner_outputs = None, encoded_concept_inputs = None, encoded_concept_outputs = None ):
        self.Print_Log( "DataLoader::Write_Formatted_File() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Fetches Concept Unique Identifier (CUI) Data From The File

        Inputs:
            file_path : file path (String)

        Outputs:
            data_list : File Data By Line As Each List Element (List)
    """
    def Read_Data( self, file_path ):
        raise NotImplementedError

    """
        Removes And Converts Special Unicode Characters From Text

        NOTE: Spaces Between Conversions Are Preserved To Maintain BioC Offset and Length Indices

        Inputs:
            text : Input String

        Outputs:
            text : Cleaned Output String
    """
    def Clean_Text( self, text ):
        # Critical Conversions/Removals
        text = text.replace( "&lt;", "   <" )           # Convert Less Than Symbol To "<" Character Symbol
        text = text.replace( "&amp;", "    &" )         # Convert Ampersand Than Symbol To "&" Character Symbol
        text = text.replace( "&gt;", "   >" )           # Convert Greater Than Symbol To ">" Character Symbol
        text = text.replace(  "&quot;", "     \"" )     # Convert Quote Symbol To '"' Character Symbol
        text = text.replace( "\[Xx][Aa][Dd]", "-" )     # Replace Unicode 'Soft Hyphen' Encoding With Regular Hyphen
        text = text.replace( "\[Uu]00[Aa][Dd]", "-" )   # Replace Unicode 'Soft Hyphen' Encoding With Regular Hyphen
        text = text.replace( "\N{SOFT HYPHEN}", "-" )   # Replace Unicode 'Soft Hyphen' Encoding With Regular Hyphen
        text = text.replace( "\u2009", " " )            # Replace Unicode 'Thin Whitespace' Encoding With Regular Whitespace
        text = text.replace( "\u00A0", " ")             # Replace Unicode 'Non-Breaking Whitespace' With Regular Whitespace
        text = text.replace( "\u2007", " ")             # Replace Unicode 'Non-Breaking Whitespace' With Regular Whitespace
        text = text.replace( "\u202F", " ")             # Replace Unicode 'Non-Breaking Whitespace' With Regular Whitespace
        text = text.replace( u'\xa0', u' ' )            # Replace Unicode 'Non-Break Space' With Regular Space

        # Less Important Conversions/Removals
        text = text.replace( "\u2236", ":" )            # Replace Unicode Ration Character ':' With Semi-colon
        text = text.replace( "\u2122", " " )            # Remove Unicode Trademark '™' Symbol
        text = text.replace( "\u2120", " " )            # Remove Unicode Service Mark '℠' Symbol
        text = text.replace( "\u2117", " " )            # Remove Unicode Sound Recording Copyright '℗' Symbol
        text = text.replace( "\u00AE", " " )            # Remove Unicode Registered '®' Symbol
        text = text.replace( "\u00A9", " " )            # Remove Unicode Copyright '©' Symbol
        text = text.replace( "\u002B", "+" )            # Replace Unicode Mathematical '+' Character With Regular +
        text = text.replace( "\u2212", "-" )            # Replace Unicode Mathematical '-' Character With Regular - (Hyphen)
        text = text.replace( "\u00D7", "x" )            # Replace Unicode Mathematical 'x' Character With Regular x
        text = text.replace( "\u22EF", "-" )            # Replace Unicode Horizontal Elipse '⋯' Character To Hyphen
        text = text.replace( "\u223C", "~" )            # Replace Unicode Tilde '~' Character With Regular ~
        text = text.replace( "\u03BA", "k")             # Replace Unicode 'Kappa' Encoding With Regular 'k'
        text = text.replace( "\u0026", "&")             # Replace Unicode '&' Encoding With Regular '&'
        text = text.replace( "\u2022", " " )            # Remove Unicode Bullet '•' Encoding
        text = text.replace( "\u25E6", " " )            # Remove Unicode Bullet '◦' Encoding
        text = text.replace( "\u2219", " " )            # Remove Unicode Bullet '∙' Encoding
        text = text.replace( "\u2023", " " )            # Remove Unicode Bullet '‣' Encoding
        text = text.replace( "\u2043", " " )            # Remove Unicode Bullet '⁃' Encoding
        text = text.replace( "\u00B0", " " )            # Remove Unicode Degree '°' Encoding
        text = text.replace( "\u221E", " " )            # Remove Unicode Bullet '∞' Encoding
        text = text.replace( "\u00B6", " " )            # Remove Unicode Paragraph '¶' Encoding
        text = text.replace( "\u00F0", "D" )
        text = text.replace( "\u00D0", "D" )

        # Remove Non-Standard Whitespace Characters
        # text = " ".join( text.split() )

        # Remove Preceding And Trailing Whitespace
        text = re.sub( r'^\s+|\s+$', "", text )

        return text

    """
        Separate Sentences/Sequences By [SEP] Token. (Used For BERT Models)

        Inputs:
            text_sequence: Text Sequences To Process (String)

        Outputs:
            text_sequence: Processed Text Sequences (String)
    """
    def Separate_Sequences_With_SEP_Token( self, text_sequence ):
        # Locate Sentence Delimiters Within The Text Sequence
        matched_sequence_delimiters = re.findall( r'(\.+\s+)|(\!+\s+)|(\?+\s+)', text_sequence )

        # Aggregate A Unique List Of Sentence Delimiters Found
        matched_sequence_delimiters = list( set( [ sequence_delimiter for sequence_delimiter_list in matched_sequence_delimiters
                                                 for sequence_delimiter in sequence_delimiter_list if sequence_delimiter != "" ] ) )

        # Add [SEP] Token After Sequence Delimiters
        for sequence_delimiter in matched_sequence_delimiters:
            text_sequence = re.sub( re.escape( sequence_delimiter ), str( sequence_delimiter ) + " [SEP] ", text_sequence )

        # Ensure Double [SEP] Tokens Do Not Exist After Regular Expression
        text_sequence     = re.sub( r'\s+\[SEP\]\s+\[SEP\]\s+', " [SEP] ", text_sequence )

        # Remove Double Whitespace Before [SEP] Token
        text_sequence     = re.sub( r'\s+\[SEP\]', " [SEP]", text_sequence )

        return text_sequence

    """
        Extracts 'x' Number Of Preceedings Sequences Given A Sequence Of Interest In Addition To
           Extracting 'y' Number Of Succeeding Sequences Given The Same Sequence Of Interest.

        Uses 'token_idx_of_interest' Which Designates The Sequences Of Interest. This Can Be Any
           Token Within The Sequence.

        NOTE: Requires Either Sequence Delimiters To Be Specified Or Uses BERT '[SEP]' Token.

        Inputs:
            text_sequence                 : String Of Sequences (String)
            token_idx_of_interest         : Index Within The Sequence Of Interest (Integer)
            number_of_preceeding_sequences: Number Of Preceeding Sequences To The Left Of Our Sequence Of Interest To Capture (Integer)
            number_of_succeeding_sequences: Number Of Preceeding Sequences To The Right Of Our Sequence Of Interest To Capture (Integer)
            use_bert_sep_delimiter        : True = Assumes 'text_sequence' Sequences Are Separated By '[SEP]' Token
                                            False = Uses Sequence Delimiter List And 'self.Separate_Sequences_With_SEP_Token()' Function
                                                    To Perform Sequence Delimiting. [SEP] Tokens Will be Removed Prior To Returning 'text_sequence'
            is_character_idx              : True = 'token_idx_of_interest' Is Character Index Within 'text_sequence' String.
                                            False = 'token_idx_of_interest' Is Number Of Splits Before Token Of Interest. (Splits By Whitespace)

        Outputs:
            text_sequence                 : Extracted Sequences (String)
            term_split_index              : Index Of Term Within Extracted Sequences (Integer)

    """
    def Extract_Surrounding_Sequences( self, text_sequence, token_idx_of_interest, number_of_preceeding_sequences = 0,
                                       number_of_succeeding_sequences = 0, use_bert_sep_delimiter = True, is_character_idx = True ):
        if number_of_preceeding_sequences < 0:
            self.Print_Log( "DataLoader::Extract_Surrounding_Sequences() - Error: 'number_of_preceeding_sequences < 0'", force_print = True )
            return text_sequence
        if number_of_succeeding_sequences < 0:
            self.Print_Log( "DataLoader::Extract_Surrounding_Sequences() - Error: 'number_of_succeeding_sequences < 0'", force_print = True )
            return text_sequence

        converted_to_split_idx = False

        # Separate Sentences/Sequences With '[SEP]' Token
        #   In The Event 'use_bert_sep_delimiter == False'
        if not use_bert_sep_delimiter:
            preceeding_sequence    = text_sequence[0:token_idx_of_interest]

            # Adjust For New Token Of Interest Location / There Will Be '[SEP]' Tokens Within The Preceeding Sequence To Account For
            token_idx_of_interest  = len( self.Separate_Sequences_With_SEP_Token( text_sequence = preceeding_sequence ).split() ) if len( preceeding_sequence ) > 0 else 0

            # Adjust For Last Element Not Being A Space i.e. 'entry_term == mdma' and 'token == (mdma,'
            if len( preceeding_sequence ) > 0 and preceeding_sequence[-1] != " ": token_idx_of_interest -= 1

            # Perform Sentence/Sequence Delimiting Using '[SEP]' Token
            text_sequence          = self.Separate_Sequences_With_SEP_Token( text_sequence = text_sequence )
            converted_to_split_idx = True

        # Convert Character Index To Token Index
        if is_character_idx and not converted_to_split_idx:
            token_idx_of_interest = len( text_sequence[0:token_idx_of_interest].split() ) if token_idx_of_interest > 0 else 0

            # Adjust For Last Element Not Being A Space i.e. 'entry_term == mdma' and 'token == (mdma,'
            if len( preceeding_sequence ) > 0 and preceeding_sequence[-1] != " ": token_idx_of_interest -= 1

        # Locate Beginning Of Previous Sentence/Sequence
        sequence_tokens, start_index, end_index = text_sequence.split(), 0, 0

        # Find [SEP] Tokens In The Sub-Word Elements Prior To The Entry Term Index.
        #   When Found, Use The Second To Last [SEP] + 1 Sub-Word Index As Our Desired Starting Index To
        #   Constrain Our Sub-Word Sequence
        if token_idx_of_interest > 0:
            sep_indices = [ idx for idx, token in enumerate( sequence_tokens[ 0 : token_idx_of_interest ] ) if token == "[SEP]" ]

            # Adjust 'number_of_preceeding_sequences' Variable To Ensure We're Not Going Over (Buffer Underflow Exception)
            if number_of_preceeding_sequences > len( sep_indices ) - 1: number_of_preceeding_sequences = len( sep_indices ) - 1

            # Check
            if number_of_preceeding_sequences == -1: number_of_preceeding_sequences = 0

            start_index = 0 if len( sep_indices ) == 0 else sep_indices[-1-number_of_preceeding_sequences] + 1
        else:
            start_index = 0

        # Find Next Sequence Index End (i.e. [SEP] Token) After Our Current Sequence Of Interest.
        #   We Use This As The End Index To Determine The Bounds Of Our Entry Term Context.
        if token_idx_of_interest + 1 < len( sequence_tokens ):
            sep_indices = [ idx for idx, token in enumerate( sequence_tokens[ token_idx_of_interest : len( sequence_tokens ) ] ) if token == "[SEP]" ]

            # Adjust 'number_of_succeeding_sequences' Variable To Ensure We're Not Going Over (Buffer Overflow Exception)
            if number_of_succeeding_sequences > len( sep_indices ) - 1: number_of_succeeding_sequences = len( sep_indices ) - 1

            # Check
            if number_of_succeeding_sequences == -1: number_of_succeeding_sequences = 0

            end_index   = len( sequence_tokens ) if len( sep_indices ) == 0 else sep_indices[number_of_succeeding_sequences] + token_idx_of_interest
        else:
            end_index   = len( sequence_tokens )

        text_sequence  = " ".join( sequence_tokens[ start_index : end_index ] )

        # Remove '[SEP]' Token And Double White Spaces
        #   In The Event 'use_bert_sep_delimiter == False'
        if not use_bert_sep_delimiter:
            text_sequence = re.sub( r'\[SEP\]', "", text_sequence )
            text_sequence = re.sub( r'\s+', " ", text_sequence )

        # Adjust For New Token Index Of Interest
        token_idx_of_interest -= start_index

        return text_sequence, token_idx_of_interest


    """
        Loads Static Embeddings From File
          Expects Standard/Plain Text Vector Format

          i.e. token_a 0.1001 0.1345 ... 0.8002
               ...
               token_n 0.9355 0.1749 ... 0.6042

        Inputs:
            file_path        : file path (String)
            lowercase        : Lowercases All Read Text (Bool)
            store_embeddings : True = Keep In Memory, False = Return To User Without Storing In Memory (Boolean)
            location         : Sets Which Variable To Store Embeddings ("a" or "b")

        Outputs:
            embedding_data   : List Static Embeddings
    """
    def Load_Embeddings( self, file_path, lowercase = False, store_embeddings = True, location = "a" ):
        embedding_data = []

        # Check(s)
        if file_path == "":
            self.Print_Log( "DataLoader::Load_Embeddings() - Warning: No File Path Specified", force_print = True )
            return []

        self.Print_Log( "DataLoader::Load_Embeddings() - File: \"" + str( file_path ) + "\"" )

        if self.utils.Check_If_File_Exists( file_path ):
            self.Print_Log( "DataLoader::Load_Embeddings() - Loading Embeddings" )
            embedding_data = self.utils.Read_Data( file_path = file_path, lowercase = lowercase )
        else:
            self.Print_Log( "DataLoader::Load_Embeddings() - Error: Embedding File Not Found In Path \"" + str( file_path ) + "\"", force_print = True )
            return []

        # Check(s)
        if len( embedding_data ) == 0:
            self.Print_Log( "DataLoader::Load_Embeddings() - Error: Embedding File Contains No Data / Length == 0" )
            return []

        # Detect Number Of Embeddings And Embedding Dimensions (Word2vec Format/Header)
        number_of_embeddings = 0
        embedding_dimensions = 0
        possible_header_info = embedding_data[0]

        # Set Embedding Variables And Remove Word2vec Header From Data
        if re.match( r'^\d+\s+\d+', possible_header_info ):
            self.Print_Log( "DataLoader::Load_Embeddings() - Detected Word2vec Embedding Header" )
            header_elements      = possible_header_info.split()
            number_of_embeddings = header_elements[0]
            embedding_dimensions = header_elements[1]
            embedding_data       = embedding_data[1:]
            self.Print_Log( "                              - Number Of Reported Embeddings          : " + str( number_of_embeddings ) )
            self.Print_Log( "                              - Number Of Reported Embedding Dimensions: " + str( embedding_dimensions ) )
        else:
            self.Print_Log( "DataLoader::LoadEmbeddings() - No Word2vec Embedding Header Detected / Computing Header Info" )
            number_of_embeddings = len( embedding_data )
            embedding_dimensions = len( embedding_data[1].split() ) - 1

        self.Print_Log( "DataLoader::Load_Embeddings() - Number Of Actual Embeddings          : " + str( len( embedding_data ) ) )
        self.Print_Log( "DataLoader::Load_Embeddings() - Number Of Actual Embedding Dimensions: " + str( len( embedding_data[1].split() ) - 1 ) )

        # Store Embeddings
        if store_embeddings:
            if location.lower() == "a":
                self.embeddings_a        = embedding_data
                self.embeddings_a_loaded = True
            else:
                self.embeddings_b        = embedding_data
                self.embeddings_b_loaded = True

        self.Print_Log( "DataLoader::Load_Embeddings() - Complete" )
        return embedding_data

    """
        Loads Concept ID List Which Populates The Concept ID Dictionary
          Expects Plain Text Separated By Newline Characters

          i.e. concept_id_a
               concept_id_b
               ...
               concept_id_n

        Inputs:
            file_path  : file path (String)
            lowercase  : True = Uncased Text / Lowercase All Read Concept IDs, False = Leave Text Cased (Boolean)

        Outputs:
            True/False : Flag - Operation Completed Successfully
    """
    def Load_Concept_ID_Data( self, file_path, lowercase = True ):
        self.Print_Log( "DataLoader::Load_Concept_ID_Data() - Loading Concept ID Dictionary" )

        concept_id_list = self.utils.Read_Data( file_path = file_path, lowercase = lowercase )

        if len( concept_id_list ) == 0 or not concept_id_list:
            self.Print_Log( "DataLoader::Load_Concept_ID_Data() - Error Loading Concept ID List" )
            return False

        self.Print_Log( "DataLoader::Load_Concept_ID_Data() - Concept ID List Loaded / Building Concept ID Dictionary" )

        # Store The First Element As Padding
        if self.padding_token not in self.concept_id_dictionary:
            self.concept_id_dictionary[self.padding_token] = 0

        self.number_of_concept_tokens = len( self.concept_id_dictionary )

        # Iterate Through All Concept IDs In Concept ID List And Store Unique Elements
        for concept_id in concept_id_list:
            if lowercase: concept_id = concept_id.lower()
            if concept_id not in self.concept_id_dictionary:
                self.concept_id_dictionary[concept_id] = self.number_of_concept_tokens
                self.number_of_concept_tokens += 1

        self.Print_Log( "DataLoader::Load_Concept_ID_Data() - Complete" )

        return True

    def Load_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "DataLoader::Load_Token_ID_Key_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    def Save_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "DataLoader::Save_Token_ID_Key_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Generates IDs For Each Token Given The Following File Format
    """
    def Generate_Token_IDs( self ):
        self.Print_Log( "DataLoader::Generate_Token_IDs() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Updates IDs For Each Token Given The Following File Format
    """
    def Update_Token_IDs( self ):
        self.Print_Log( "DataLoader::Update_Token_IDs() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Saves Vectorized Model Inputs/Outputs To File.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Save_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches NER Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        self.Print_Log( "DataLoader::Get_Token_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches Concept Token ID From String.

        Inputs:
            concept    : Token (String)

        Outputs:
            concept_id : Token ID Value (Integer)
    """
    def Get_Concept_ID( self, concept ):
        self.Print_Log( "DataLoader::Get_Concept_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches NER Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Token String (String)
    """
    def Get_Token_From_ID( self, index_value ):
        self.Print_Log( "DataLoader::Get_Token_From_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches NER Label From ID Value.

        Inputs:
            label_id  : Token (String)

        Outputs:
            ner_label : Token ID Value (Integer)
    """
    def Get_NER_Label_From_ID( self, label_id ):
        self.Print_Log( "DataLoader::Get_NER_Label_From_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches Concept Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Concept Token String (String)
    """
    def Get_Concept_From_ID( self, index_value  ):
        self.Print_Log( "DataLoader::Get_Concept_From_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Reinitialize Token ID Values For Primary And Secondary ID Dictionaries

        Inputs:
            None

        Outputs:
            None
    """
    def Reinitialize_Token_ID_Values( self ):
        self.Print_Log( "DataLoader::Reinitialize_Token_ID_Values() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Closes Read_Data() In-Line Read File Handle

        Inputs:
            None

        Outputs:
            None
    """
    def Close_Read_File_Handle( self ):
        if self.read_file_handle is not None: self.read_file_handle.close()

    """
        Clears Embedding Data From Memory

        Inputs:
            None

        Outputs:
            None
    """
    def Clear_Embedding_Data( self ):
        self.embeddings_a        = []
        self.embeddings_b        = []
        self.embeddings_a_loaded = False
        self.embeddings_b_loaded = False
        self.simulate_embeddings_a_loaded = False
        self.simulate_embeddings_b_loaded = False

    """
        Clears Data From Memory

        Inputs:
            None

        Outputs:
            None
    """
    def Clear_Data( self ):
        self.number_of_input_tokens       = 0
        self.number_of_concept_tokens     = 0
        self.max_sequence_length          = 0
        self.label_sequence_padding       = 0
        self.pad_label_token_id           = 0
        self.pad_token_segment_id         = 0
        self.sequence_a_segment_id        = 0
        self.data_list                    = []
        self.ner_inputs                   = []
        self.ner_outputs                  = []
        self.concept_inputs               = []
        self.concept_outputs              = []
        self.ner_val_inputs               = []
        self.ner_val_outputs              = []
        self.concept_val_inputs           = []
        self.concept_val_outputs          = []
        self.ner_eval_inputs              = []
        self.ner_eval_outputs             = []
        self.concept_eval_inputs          = []
        self.concept_eval_outputs         = []
        self.embeddings_a                 = []
        self.embeddings_b                 = []
        self.token_id_dictionary          = {}
        self.concept_id_dictionary        = {}
        self.concept_instance_data_idx    = []
        self.concept_frequency_dictionary = {}
        self.annotation_labels            = {}
        self.oov_term_dict                = {}
        self.ignore_label_type_list       = []
        self.cl_accepted_labels           = []
        self.padding_token                = ""
        self.cui_less_token               = ""
        self.sub_word_cls_token           = ""
        self.sub_word_sep_token           = ""
        self.sub_word_pad_token           = ""
        self.embeddings_a_loaded          = False
        self.embeddings_b_loaded          = False
        self.lowercase_text               = False
        self.simulate_embeddings_a_loaded = False
        self.simulate_embeddings_b_loaded = False
        self.read_file_handle             = None
        self.generated_embedding_ids      = False
        self.skip_out_of_vocabulary_words = False
        self.skip_individual_mentions     = False
        self.skip_composite_mentions      = False
        self.max_sequence_length_set      = False


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Prints Debug Text To Console
    """
    def Print_Log( self, text, print_new_line = True, force_print = False ):
        if self.debug_log or force_print:
            print( text ) if print_new_line else print( text, end = " " )
        if self.write_log:
            self.Write_Log( text, print_new_line )

    """
        Prints Debug Log Text To File
    """
    def Write_Log( self, text, print_new_line = True ):
        if self.write_log and self.debug_log_file_handle is not None:
            self.debug_log_file_handle.write( text + "\n" ) if print_new_line else self.debug_log_file_handle.write( text )


    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################

    def Get_Embeddings_A( self ):                   return self.embeddings_a

    def Get_Embeddings_B( self ):                   return self.embeddings_b

    def Get_Concept_Frequency_Dictionary( self ):   return self.concept_frequency_dictionary

    def Get_Token_ID_Dictionary( self ):            return self.token_id_dictionary

    def Get_Concept_ID_Dictionary( self ):          return self.concept_id_dictionary

    def Get_Concept_Instance_Data_IDX( self ):      return self.concept_instance_data_idx

    def Get_Annotation_Labels( self ):              return self.annotation_labels

    def Get_Number_Of_Unique_Tokens( self ):        return len( self.token_id_dictionary )

    def Get_Number_Of_Unique_Concepts( self ):      return len( self.concept_id_dictionary )

    def Get_Number_Of_Input_Elements( self ):       return self.number_of_input_tokens

    def Get_Number_Of_Concept_Elements( self ):     return self.number_of_concept_tokens

    def Get_Number_Of_Annotation_Labels( self ):    return len( self.annotation_labels )

    def Get_Max_Sequence_Length( self ):            return self.max_sequence_length

    def Get_Label_Sequence_Padding( self ):         return self.label_sequence_padding

    def Get_Pad_Label_Token_ID( self ):             return self.pad_label_token_id

    def Get_Pad_Token_Segment_ID( self ):           return self.pad_token_segment_id

    def Get_Sequence_A_Segment_ID( self ):          return self.sequence_a_segment_id

    def Get_Skip_Out_Of_Vocabulary_Words( self ):   return self.skip_out_of_vocabulary_words

    def Get_Skip_Individual_Mentions( self ):       return self.skip_individual_mentions

    def Get_Skip_Composite_Mentions( self ):        return self.skip_composite_mentions

    def Get_Ignore_Label_Type_List( self ):         return self.ignore_label_type_list

    def Get_Data( self ):                           return self.data_list

    def Get_NER_Inputs( self ):                     return self.ner_inputs

    def Get_NER_Outputs( self ):                    return self.ner_outputs

    def Get_NER_Validation_Inputs( self ):          return self.ner_val_inputs

    def Get_NER_Validation_Outputs( self ):         return self.ner_val_outputs

    def Get_NER_Evaluation_Inputs( self ):          return self.ner_eval_inputs

    def Get_NER_Evaluation_Outputs( self ):         return self.ner_eval_outputs

    def Get_Concept_Inputs( self ):                 return self.concept_inputs

    def Get_Concept_Outputs( self ):                return self.concept_outputs

    def Get_Concept_Validation_Inputs( self ):      return self.concept_val_inputs

    def Get_Concept_Validation_Outputs( self ):     return self.concept_val_outputs

    def Get_Concept_Evaluation_Inputs( self ):      return self.concept_eval_inputs

    def Get_Concept_Evaluation_Outputs( self ):     return self.concept_eval_outputs

    def Get_Number_Of_Embeddings_A( self ):         return len( self.embeddings_a )

    def Get_Number_Of_Embeddings_B( self ):         return len( self.embeddings_a )

    def Get_OOV_Term_Dict( self ):                  return self.oov_term_dict

    def Get_Number_Of_OOV_Terms( self ):            return len( self.oov_term_dict.keys() )

    def Get_Total_OOV_Term_Frequency( self ):       return sum( list( self.oov_term_dict.values() ) ) if len( self.oov_term_dict.keys() ) > 0 else 0

    # Note: Call 'Generate_Token_IDs()' Prior To Calling This Function Or Subtract '1' From The Return Value
    def Get_Embedding_Dimension_Size( self ):       return len( self.embeddings_a[1] ) if len( self.embeddings_a ) > 0 else 0

    def Is_Embeddings_A_Loaded( self ):             return self.embeddings_a_loaded

    def Is_Embeddings_B_Loaded( self ):             return self.embeddings_b_loaded

    def Simulate_Embeddings_A_Loaded_Mode( self ):  return self.simulate_embeddings_a_loaded

    def Simulate_Embeddings_B_Loaded_Mode( self ):  return self.simulate_embeddings_b_loaded

    def Is_Data_Loaded( self ):                     return True if len( self.Get_Data() ) > 0 else False

    def Is_Dictionary_Loaded( self ):               return True if self.Get_Number_Of_Unique_Tokens() > 0 else False

    def Get_Padding_Token( self ):                  return self.padding_token

    def Get_CUI_LESS_Token( self ):                 return self.cui_less_token

    def Get_Sub_Word_CLS_Token( self ):             return self.sub_word_cls_token

    def Get_Sub_Word_SEP_Token( self ):             return self.sub_word_sep_token

    def Get_Sub_Word_PAD_Token( self ):             return self.sub_word_pad_token

    def Get_Sub_Word_CLS_Token_ID( self ):          return self.sub_word_cls_token_id

    def Get_Sub_Word_SEP_Token_ID( self ):          return self.sub_word_sep_token_id

    def Get_Sub_Word_PAD_Token_ID( self ):          return self.sub_word_pad_token_id

    def Get_CL_Accepted_Labels( self ):             return self.cl_accepted_labels

    def Get_Version( self ):                        return self.version

    # Note: Call 'Generate_Token_IDs()' Prior To Calling This Function Or Subtract '1' From The Return Value
    def Get_Embedding_A_Dimension( self ):
        # Embeddings Are Strings
        if isinstance( self.embeddings_a[1], str ):
            return len( self.embeddings_a[1].split() ) if len( self.embeddings_a ) > 0 else -1
        # Embeddings Are Numpy Arrays
        else:
            return self.embeddings_a.shape[1] if self.embeddings_a.shape[0] > 0 else -1

    # Note: Call 'Generate_Token_IDs()' Prior To Calling This Function Or Subtract '1' From The Return Value
    def Get_Embedding_B_Dimension( self ):
        # Embeddings Are Strings
        if isinstance( self.embeddings_b[1], str ):
            return len( self.embeddings_b[1].split() ) if len( self.embeddings_b ) > 0 else -1
        # Embeddings Are Numpy Arrays
        else:
            return self.embeddings_b.shape[1] if self.embeddings_b.shape[0] > 0 else -1


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################

    def Set_Embeddings_A( self, embeddings ):               self.embeddings_a = embeddings

    def Set_Embeddings_B( self, embeddings ):               self.embeddings_b = embeddings

    def Set_NER_Inputs( self, inputs ):                     self.ner_inputs = inputs

    def Set_NER_Outputs( self, outputs ):                   self.ner_outputs = outputs

    def Set_NER_Validation_Inputs( self, inputs ):          self.ner_val_inputs = inputs

    def Set_NER_Validation_Outputs( self, outputs ):        self.ner_val_outputs = outputs

    def Set_NER_Evaluation_Inputs( self, inputs ):          self.ner_eval_inputs = inputs

    def Set_NER_Evaluation_Outputs( self, outputs ):        self.ner_eval_outputs = outputs

    def Set_Concept_Inputs( self, inputs ):                 self.concept_inputs = inputs

    def Set_Concept_Outputs( self, outputs ):               self.concept_outputs = outputs

    def Set_Concept_Validation_Inputs( self, inputs ):      self.concept_val_inputs = inputs

    def Set_Concept_Validation_Outputs( self, outputs ):    self.concept_val_outputs = outputs

    def Set_Concept_Evaluation_Inputs( self, inputs ):      self.concept_eval_inputs = inputs

    def Set_Concept_Evaluation_Outputs( self, outputs ):    self.concept_eval_outputs = outputs

    def Set_Max_Sequence_Length( self, value ):             self.max_sequence_length = value

    def Set_Token_ID_Dictionary( self, id_dictionary ):     self.token_id_dictionary = id_dictionary

    def Set_Concept_ID_Dictionary( self, concept_dict ):    self.concept_id_dictionary = concept_dict

    def Set_Concept_Instance_Data_IDX( self, id_list ):     self.concept_instance_data_idx = id_list

    def Set_Debug_Log_File_Handle( self, file_handle ):     self.debug_log_file_handle = file_handle

    def Set_Skip_Individual_Mentions( self, value ):        self.skip_individual_mentions = value

    def Set_Skip_Composite_Mention( self, value ):          self.skip_composite_mentions = value

    def Set_Simulate_Embeddings_A_Loaded_Mode( self, value ):
        self.simulate_embeddings_a_loaded  = value
        self.generated_embedding_ids       = value
        if value: self.embeddings_a_loaded = value

    def Set_Simulate_Embeddings_B_Loaded_Mode( self, value ):
        self.simulate_embeddings_b_loaded  = value
        self.generated_embedding_ids       = value
        if value: self.embeddings_b_loaded = value

    def Get_Token_Embedding_A( self, token ):
        if token in self.token_id_dictionary: return self.embeddings_a[self.token_id_dictionary[token]]
        return []

    def Get_Token_Embedding_B( self, token ):
        if token in self.concept_id_dictionary: return self.embeddings_b[self.concept_id_dictionary[token]]
        return []


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from models import DataLoader\n" )
    print( "     data_loader = DataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()

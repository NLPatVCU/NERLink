#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/08/2021                                                                   #
#    Revised: 11/16/2022                                                                   #
#                                                                                          #
#    Main Driver Class For The NERLink Package.                                            #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import os, re, time
import numpy as np
import matplotlib.pyplot as plt
from sparse       import COO
from scipy.sparse import csr_matrix

# Custom Modules
from NERLink.DataLoader.BioCreative import *
from NERLink.DataLoader.Misc        import *
from NERLink.Models.NER             import *
from NERLink.Models.ConceptLinking  import *
from NERLink.Models.MultiTaskModels import *
from NERLink.Misc                   import Utils


############################################################################################
#                                                                                          #
#    NERLink Model Interface Class                                                         #
#                                                                                          #
############################################################################################

class NERLink:
    """
    """
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "ner_bilstm", optimizer = 'adam',
                  activation_function = 'sigmoid', loss_function = "binary_crossentropy", margin = 30.0, scale = 0.35,
                  bilstm_merge_mode = "concat", bilstm_dimension_size = 64, learning_rate = 0.005, epochs = 30, momentum = 0.05,
                  dropout = 0.1, batch_size = 16, prediction_threshold = 0.5, shuffle = True, skip_out_of_vocabulary_words = True,
                  use_csr_format = True, per_epoch_saving = True, use_gpu = True, device_name = "/gpu:0", verbose = 2,
                  enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_monitor = "val_loss",
                  early_stopping_patience = 3, use_batch_normalization = False, checkpoint_directory = "./ckpt_models",
                  trainable_weights = False, embedding_a_path = "", embedding_b_path = "", final_layer_type = "dense",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, model_path = "", output_embedding_type = "",
                  skip_composite_mention = False, skip_individual_mention = False, lowercase = False, ignore_label_type_list = [],
                  class_weights = None, sample_weights = None, use_cosine_annealing = False, cosine_annealing_min = 1e-6,
                  cosine_annealing_max = 2e-4 ):
        self.version                       = 0.24
        self.model                         = None                            # Automatically Set After Calling 'NERLink::Build_Model()' Function
        self.debug_log                     = print_debug_log                 # Options: True, False
        self.write_log                     = write_log_to_file               # Options: True, False
        self.debug_log_file_handle         = None                            # Debug Log File Handle
        self.checkpoint_directory          = checkpoint_directory            # Path (String)
        self.model_data_prepared           = False                           # Options: True, False (Default: False)
        self.data_loader                   = None
        self.is_ner_model                  = False
        self.is_concept_linking_model      = False
        self.is_ner_cl_multi_task_model    = False
        self.debug_log_file_name           = "NERLink_Log.txt"               # File Name (String)
        self.model_types                   = ["ner_bilstm", "ner_elmo", "ner_bert", "concept_linking",
                                              "concept_linking_bert", "concept_linking_bert_distributed",
                                              "concept_linking_embedding_similarity", "concept_linking_bert_embedding_similarity",
                                              "ner_concept_linking_multi_task_bert"]

        # Create Log File Handle
        if self.write_log and self.debug_log_file_handle is None:
            self.debug_log_file_handle = open( self.debug_log_file_name, "w" )

        # Check BERT-Specific Default Settings / Model Path
        if model_path == "" and "bert" in network_model:
            self.Print_Log( "NERLink::__init__() - Warning: BERT Model Path Not Specified / Setting BERT Model As 'bert-base-cased'", force_print = True )
            model_path = "bert-base-cased"

        # ----------------------------------------------- #
        #  Create New DataLoader Instance With Options    #
        #  Model Specific DataLoader Parameters/Settings  #
        # ----------------------------------------------- #
        if network_model in ["ner_bilstm", "ner_elmo", "concept_linking"]:
            self.data_loader = BioCreativeDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                      skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                      debug_log_file_handle = self.debug_log_file_handle, skip_composite_mentions = skip_composite_mention,
                                                      skip_individual_mentions = skip_individual_mention, ignore_label_type_list = ignore_label_type_list )
        elif network_model in ["ner_bert", "concept_linking_bert"]:
            self.data_loader = BERTBioCreativeDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                          debug_log_file_handle = self.debug_log_file_handle, bert_model = model_path,
                                                          skip_composite_mentions = skip_composite_mention, skip_individual_mentions = skip_individual_mention,
                                                          ignore_label_type_list = ignore_label_type_list )
        elif network_model in ["concept_linking_bert_distributed"]:
            self.data_loader = BERTDistributedBioCreativeDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                                     skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                                     debug_log_file_handle = self.debug_log_file_handle, bert_model = model_path,
                                                                     skip_composite_mentions = skip_composite_mention, skip_individual_mentions = skip_individual_mention,
                                                                     ignore_label_type_list = ignore_label_type_list )
        elif network_model in ["concept_linking_embedding_similarity"]:
            self.data_loader = MLPSimilarityDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                        skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                        debug_log_file_handle = self.debug_log_file_handle, skip_composite_mentions = skip_composite_mention,
                                                        skip_individual_mentions = skip_individual_mention, ignore_label_type_list = ignore_label_type_list )
        elif network_model in ["concept_linking_bert_embedding_similarity"]:
            self.data_loader = BERTSimilarityDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                         skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                         debug_log_file_handle = self.debug_log_file_handle, skip_composite_mentions = skip_composite_mention,
                                                         skip_individual_mentions = skip_individual_mention, ignore_label_type_list = ignore_label_type_list )
        elif network_model in ["ner_concept_linking_multi_task_bert"]:
            self.data_loader = BERTBioCreativeMultiTaskDataLoader( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file,
                                                                   skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, lowercase = lowercase,
                                                                   debug_log_file_handle = self.debug_log_file_handle, skip_composite_mentions = skip_composite_mention,
                                                                   skip_individual_mentions = skip_individual_mention, ignore_label_type_list = ignore_label_type_list )
        else:
            self.Print_Log( "NERLink::Init() - Error Model \"" + str( network_model ) + "\"'s DataLoader Not Implemented", force_print = True )
            raise NotImplementedError

        # Create New Utils Instance
        self.utils = Utils()

        self.Print_Log( "NERLink::Init() - Current Working Directory: \"" + str( self.utils.Get_Working_Directory() ) + "\"" )

        # Check(s)
        if network_model not in self.model_types:
            self.Print_Log( "NERLink::Init() - Warning: Network Model Type Is Not 'ner_bilstm', 'ner_bert', 'ner_elmo', 'concept_linking'",    force_print = True )
            self.Print_Log( "                                                     'concept_linking_bert', 'concept_linking_bert_distributed'", force_print = True )
            self.Print_Log( "                                                     'concept_linking_embedding_similarity', 'concept_linking_bert_embedding_similarity'", force_print = True )
            self.Print_Log( "                - Resetting Network Model Type To: 'ner_bilstm'", force_print = True )
            network_model  = "ner_bilstm"
            continue_query = input( "Continue? (Y/N)\n" )
            if re.search( r"[Nn]", continue_query ): exit()
        else:
            self.Print_Log( "NERLink::Init() - Network Model: \"" + str( network_model ) + "\"", force_print = True )

        if use_csr_format == False:
            self.Print_Log( "NERLink::Init() - Warning: Use CSR Mode = False / High Memory Consumption May Occur When Vectorizing Data-Sets", force_print = True )
        else:
            self.Print_Log( "NERLink::Init() - Using CSR Matrix Format" )

        if per_epoch_saving:
            self.Create_Checkpoint_Directory()

        # Create LBD Model Type
        if network_model == "ner_bilstm":
            self.is_ner_model = True
            self.model = BiLSTMModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, verbose = verbose,
                                      optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                      learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                      prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                      per_epoch_saving = per_epoch_saving,  bilstm_merge_mode = bilstm_merge_mode, bilstm_dimension_size = bilstm_dimension_size,
                                      device_name = device_name, debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                      enable_early_stopping = enable_early_stopping, early_stopping_monitor = early_stopping_monitor,
                                      early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                      trainable_weights = trainable_weights, embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path,
                                      final_layer_type = final_layer_type, feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay,
                                      class_weights = class_weights, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                      cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "ner_elmo":
            self.is_ner_model = True
            if model_path == "": model_path = "https://tfhub.dev/google/elmo/2"

            self.model = ELMoModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, verbose = verbose,
                                    optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                    learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                    prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                    per_epoch_saving = per_epoch_saving,  bilstm_merge_mode = bilstm_merge_mode, bilstm_dimension_size = bilstm_dimension_size,
                                    device_name = device_name, debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                    enable_early_stopping = enable_early_stopping, early_stopping_monitor = early_stopping_monitor,
                                    early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                    trainable_weights = trainable_weights, embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path,
                                    final_layer_type = final_layer_type, feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay,
                                    model_path = model_path, class_weights = class_weights, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                    cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "ner_bert":
            self.is_ner_model = True

            # Check
            if use_csr_format  : use_csr_format = False

            self.model = BERTModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, verbose = verbose,
                                    optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                    learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                    prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                    per_epoch_saving = per_epoch_saving,  bilstm_merge_mode = bilstm_merge_mode, bilstm_dimension_size = bilstm_dimension_size,
                                    device_name = device_name, debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                    enable_early_stopping = enable_early_stopping, early_stopping_monitor = early_stopping_monitor,
                                    early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                    trainable_weights = trainable_weights, embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path,
                                    final_layer_type = final_layer_type, feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay,
                                    model_path = model_path, class_weights = class_weights, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                    cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "concept_linking":
            self.is_concept_linking_model = True
            self.model = CLMLPModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                     optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                     learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                     prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                     per_epoch_saving = per_epoch_saving, device_name = device_name, debug_log_file_handle = self.debug_log_file_handle,
                                     enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                                     early_stopping_monitor = early_stopping_monitor, early_stopping_patience = early_stopping_patience,
                                     use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_a_path = embedding_a_path,
                                     embedding_b_path = embedding_b_path,  verbose = verbose, final_layer_type = final_layer_type, feature_scale_value = feature_scale_value,
                                     learning_rate_decay = learning_rate_decay, class_weights = class_weights, sample_weights = sample_weights,
                                     use_cosine_annealing = use_cosine_annealing, cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "concept_linking_bert":
            self.is_concept_linking_model = True

            # Check BERT-Specific Default Settings
            if output_embedding_type == "":
                self.Print_Log( "NERLink::__init__() - Warning: BERT Output Embedding Type Not Specified / Using 'average' As Default Setting", force_print = True )
                output_embedding_type = "average"

            self.model = CLBERTModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                      optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                      learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                      prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                      per_epoch_saving = per_epoch_saving, device_name = device_name, debug_log_file_handle = self.debug_log_file_handle,
                                      enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                                      early_stopping_monitor = early_stopping_monitor, early_stopping_patience = early_stopping_patience,
                                      use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_a_path = embedding_a_path,
                                      embedding_b_path = embedding_b_path, verbose = verbose, final_layer_type = final_layer_type, class_weights = class_weights,
                                      feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, model_path = model_path,
                                      output_embedding_type = output_embedding_type, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                      cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "concept_linking_bert_distributed":
            self.is_concept_linking_model = True
            self.model = CLBERTModelDistributed( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                                 optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                                 learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                                 prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu,
                                                 per_epoch_saving = per_epoch_saving, device_name = device_name, debug_log_file_handle = self.debug_log_file_handle,
                                                 enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                                                 early_stopping_monitor = early_stopping_monitor, early_stopping_patience = early_stopping_patience,
                                                 use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_a_path = embedding_a_path,
                                                 embedding_b_path = embedding_b_path, verbose = verbose, final_layer_type = final_layer_type, feature_scale_value = feature_scale_value,
                                                 class_weights = class_weights, learning_rate_decay = learning_rate_decay, model_path = model_path,
                                                 output_embedding_type = output_embedding_type, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                                 cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "concept_linking_embedding_similarity":
            self.is_concept_linking_model = True
            self.model = CLMLPSimilarityModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                               optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                               learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                               shuffle = shuffle, use_gpu = use_gpu, per_epoch_saving = per_epoch_saving, device_name = device_name,
                                               debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                               enable_early_stopping = enable_early_stopping, scale = scale, early_stopping_monitor = early_stopping_monitor,
                                               early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                               embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path, verbose = verbose, feature_scale_value = feature_scale_value,
                                               class_weights = class_weights, learning_rate_decay = learning_rate_decay, sample_weights = sample_weights,
                                               use_cosine_annealing = use_cosine_annealing, cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "concept_linking_bert_embedding_similarity":
            self.is_concept_linking_model = True

            # Check BERT-Specific Default Settings
            if output_embedding_type == "":
                self.Print_Log( "NERLink::__init__() - Warning: BERT Output Embedding Type Not Specified / Using 'average' As Default Setting", force_print = True )
                output_embedding_type = "average"

            self.model = CLBERTSimilarityModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                                optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                                learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                                shuffle = shuffle, use_gpu = use_gpu, per_epoch_saving = per_epoch_saving, device_name = device_name,
                                                debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                                enable_early_stopping = enable_early_stopping, scale = scale, early_stopping_monitor = early_stopping_monitor,
                                                early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                                embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path, verbose = verbose, feature_scale_value = feature_scale_value,
                                                class_weights = class_weights, learning_rate_decay = learning_rate_decay, trainable_weights = trainable_weights,
                                                sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                                cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        elif network_model == "ner_concept_linking_multi_task_bert":
            self.is_ner_cl_multi_task_model = True
            self.model = NERCLMultiTaskModel( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, margin = margin,
                                              optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,
                                              learning_rate = learning_rate, epochs = epochs, momentum = momentum, dropout = dropout, batch_size = batch_size,
                                              shuffle = shuffle, use_csr_format = use_csr_format, use_gpu = use_gpu, per_epoch_saving = per_epoch_saving,
                                              device_name = device_name, debug_log_file_handle = self.debug_log_file_handle, enable_tensorboard_logs = enable_tensorboard_logs,
                                              enable_early_stopping = enable_early_stopping, scale = scale, early_stopping_monitor = early_stopping_monitor,
                                              early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                                              embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path, verbose = verbose, feature_scale_value = feature_scale_value,
                                              class_weights = class_weights, learning_rate_decay = learning_rate_decay, trainable_weights = trainable_weights,
                                              sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                              cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        else:
            self.Print_Log( "NERLink::__init__() - Error: Model \'" + str( network_model ) + "\' Not Implemented", force_print = True )
            raise NotImplementedError

    """
       Remove Variables From Memory
    """
    def __del__( self ):
        del self.model
        del self.utils
        del self.data_loader

        if self.write_log and self.debug_log_file_handle is not None: self.debug_log_file_handle.close()

    """
       Prepares Model And Data For Training/Testing
           Current Neural Architecture Implementations: BiLSTM, ELMo, BERT For NER & MLP, CosFace, ArcFace & SphereFace For Concept Linking
    """
    def Prepare_Model_Data( self, data_file_path = "", val_file_path = "", eval_file_path = "", data_instances = [],
                            embedding_dimension_size = 200, lowercase = True, term_sequence_only = False, concept_delimiter = ",",
                            mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = False,
                            force_run = False, generate_token_ids = True, skip_building_model = False, ignore_output_errors = False ):
        if self.Is_Model_Data_Prepared() and force_run == False:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Warning: Model Data Has Already Been Prepared" )
            return True

        # Bug Fix: User Enabled 'per_epoch_saving' After Initially Disabled In __init__()
        if self.model.Get_Per_Epoch_Saving(): self.Create_Checkpoint_Directory()

        #######################################################################
        #                                                                     #
        #   Prepare Embeddings & Data-set                                     #
        #                                                                     #
        #######################################################################

        data_loader, model_data, model_val_data, model_eval_data = self.Get_Data_Loader(), None, None, None

        # Load Embeddings A
        if self.model.Get_Embedding_A_Path() != "" and data_loader.Is_Embeddings_A_Loaded() == False and force_run == False:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Loading Embeddings: " + str( self.model.Get_Embedding_A_Path() ), force_print = True )
            embeddings = data_loader.Load_Embeddings( self.model.Get_Embedding_A_Path(), lowercase = lowercase )

            # Check
            if isinstance( embeddings, list ) and len( embeddings ) == 0 or isinstance( embeddings, np.ndarray ) and embeddings.shape[0] == 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Error Loading Embedding A", force_print = True )
                self.Print_Log( "NERLink::Prepare_Model_Data() -       Embedding A Path: " + str( self.model.Get_Embedding_A_Path() ), force_print = True )
                return False
            else:
                self.model.Set_Embeddings_A_Loaded( data_loader.Is_Embeddings_A_Loaded() )

        # Load Embeddings B
        if self.model.Get_Embedding_B_Path() != "" and data_loader.Is_Embeddings_B_Loaded() == False and force_run == False:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Loading Embeddings: " + str( self.model.Get_Embedding_B_Path() ), force_print = True )
            embeddings = data_loader.Load_Embeddings( self.model.Get_Embedding_B_Path(), lowercase = lowercase, location = "b" )

            # Check
            if isinstance( embeddings, list ) and len( embeddings ) == 0 or isinstance( embeddings, np.ndarray ) and embeddings.shape[0] == 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Error Loading Embedding B", force_print = True )
                self.Print_Log( "NERLink::Prepare_Model_Data() -       Embedding B Path: " + str( self.model.Get_Embedding_B_Path() ), force_print = True )
                return False
            else:
                self.model.Set_Embeddings_B_Loaded( data_loader.Is_Embeddings_B_Loaded() )

        # Read Training Data From File Using 'data_file_path'
        #   Load From A File
        if data_file_path != "":
            self.Print_Log( "NERLink::Prepare_Model_Data() - Reading Training Data: " + str( data_file_path ), force_print = True )
            model_data = data_loader.Read_Data( file_path = data_file_path, lowercase = lowercase )

            # Generate Token IDs
            # NOTE: This Is Skipped When Loading A Pre-Trained Model For Inference Or Further Fine-Tuning
            if generate_token_ids:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Generating Token IDs From Data", force_print = True )
                data_loader.Generate_Token_IDs( lowercase = lowercase )

        # Train Using Passed Data From 'data_instances' List Parameter.
        #   This Also Assumes Token ID Dictionary Has Been Previously Generated.
        else:
            model_data = data_instances

            # Generate Token IDs
            # NOTE: This Is Skipped When Loading A Pre-Trained Model For Inference Or Further Fine-Tuning
            if generate_token_ids:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Generating Token IDs From Data", force_print = True )
                data_loader.Generate_Token_IDs( data_list = model_data, lowercase = lowercase )

        # Read Data From File Using 'val_file_path'
        #   Load From A File
        if val_file_path != "":
            self.Print_Log( "NERLink::Prepare_Model_Data() - Reading Validation Data: " + str( val_file_path ), force_print = True )
            model_val_data = data_loader.Read_Data( file_path = val_file_path, lowercase = lowercase, keep_in_memory = False )

            # Generate Token IDs
            # NOTE: This Is Skipped When Loading A Pre-Trained Model For Inference Or Further Fine-Tuning
            if generate_token_ids:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Generating Token IDs From Validation Data", force_print = True )
                data_loader.Generate_Token_IDs( data_list = model_val_data, lowercase = lowercase, update_dict = True )
        else:
            model_val_data = []

        # Read Data From File Using 'eval_file_path'
        #   Load From A File
        if eval_file_path != "":
            self.Print_Log( "NERLink::Prepare_Model_Data() - Reading Evaluation Data: " + str( eval_file_path ), force_print = True )
            model_eval_data = data_loader.Read_Data( file_path = eval_file_path, lowercase = lowercase, keep_in_memory = False )

            # Generate Token IDs
            # NOTE: This Is Skipped When Loading A Pre-Trained Model For Inference Or Further Fine-Tuning
            if generate_token_ids:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Generating Token IDs From Evaluation Data", force_print = True )
                data_loader.Generate_Token_IDs( data_list = model_eval_data, lowercase = lowercase, update_dict = True )
        else:
            model_eval_data = []

        embeddings = []

        # Assuming We're Actually Using Embeddings, Fetch Them From The DataLoader And Pass To The Model.
        #    If We're Refining A Model, It Already Contains The Embedding Representations.
        #    So We Just Set 'simulate_embeddings_loaded' Parameter To 'True' In The DataLoader Class To Avoid Generating Them Again.
        if data_loader.Is_Embeddings_A_Loaded() and data_loader.Simulate_Embeddings_A_Loaded_Mode() == False:
            embeddings = data_loader.Get_Embeddings_A()

            if len( embeddings ) == 0: self.Print_Log( "NERLink::Prepare_Model_Data() - Warning: Embeddings Data Length == 0" )

            embedding_dimension_size = data_loader.Get_Embedding_A_Dimension()
            self.Save_Model_Keys( model_name = "last_" + self.model.Get_Network_Model() + "_model" )

        self.Print_Log( "NERLink::Prepare_Model_Data() - Network Model Type - " + str( self.model.Get_Network_Model() ) )

        #######################################################################
        #                                                                     #
        #   Prepare Model Input/Output Data                                   #
        #                                                                     #
        #######################################################################

        # Encode Model Data
        self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding Model Input & Output Data", force_print = True )

        train_input, train_output = None, None

        # Concept Linking Model(s)
        if self.is_concept_linking_model:
            # Encode Data Instances
            #   If 'model_data' != None, Encode Data Else Encode Data Within DataLoader Class
            if len( model_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Training Data", force_print = True )
                train_input, train_output = self.Encode_CL_Model_Data( data_list = model_data, use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                       term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter,
                                                                       mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                       restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                       ignore_output_errors = ignore_output_errors )

            # Encode Validation Data
            if len( model_val_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Validation Data", force_print = True )
                val_input, val_output = self.Encode_CL_Model_Data( data_list = model_val_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_validation_data = True,
                                                                   term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter,
                                                                   mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                   restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                   ignore_output_errors = ignore_output_errors )

            # Encode Evaluation Data
            #   If 'model_data' != None, Encode Data Else Encode Data Within DataLoader Class
            if len( model_eval_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Evaluation Data", force_print = True )
                eval_input, eval_output = self.Encode_CL_Model_Data( data_list = model_eval_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_evaluation_data = True,
                                                                     term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter,
                                                                     mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                     restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                     ignore_output_errors = ignore_output_errors )
        # NER Model(s)
        elif self.is_ner_model:
            # Train On Data Instances Passed By Parameter
            #   If 'model_data' != None, Encode Data Else Encode Data Within DataLoader Class
            if len( model_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding NER Training Data", force_print = True )
                train_input, train_output = self.Encode_NER_Model_Data( data_list = model_data, use_csr_format = self.model.Get_Use_CSR_Format() )

            # Encode Validation Data
            if len( model_val_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding NER Validation Data", force_print = True )
                val_input, val_output = self.Encode_NER_Model_Data( data_list = model_val_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_validation_data = True )

            # Encode Evaluation Data
            if len( model_eval_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding NER Evaluation Data", force_print = True )
                eval_input, eval_output = self.Encode_NER_Model_Data( data_list = model_eval_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_evaluation_data = True )

            # Get String Sequences, Used For ELMo Input
            #   TODO: Create ELMo-Specific DataLoader To Maintain Code Modularization
            if self.Get_Network_Model() == "ner_elmo":
                if len( model_data ) > 0:
                    train_input = self.Tokenize_Model_Data( data_list = model_data )
                    train_input = np.asarray( train_input, dtype = object )
                    data_loader.Set_NER_Inputs( train_input )

                if len( model_val_data ) > 0:
                    val_input = self.Tokenize_Model_Data( data_list = model_val_data )
                    val_input = np.asarray( val_input, dtype = object )
                    data_loader.Set_NER_Validation_Inputs( val_input )

                if len( model_eval_data ) > 0:
                    eval_input = self.Tokenize_Model_Data( data_list = model_eval_data )
                    eval_input = np.asarray( eval_input, dtype = object )
                    data_loader.Set_NER_Evaluation_Inputs( eval_input )

        # NER-CL Mult-Task Model(s)
        elif self.is_ner_cl_multi_task_model:
            # Encode Data Instances
            #   If 'model_data' != None, Encode Data Else Encode Data Within DataLoader Class
            if len( model_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Training Data", force_print = True )
                train_input, train_output = self.Encode_Model_Data( data_list = model_data, use_csr_format = self.model.Get_Use_CSR_Format(),
                                                                    term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter,
                                                                    mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                    restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                    ignore_output_errors = ignore_output_errors, pad_output = False )

            # Encode Validation Data
            if len( model_val_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Validation Data", force_print = True )
                val_input, val_output = self.Encode_Model_Data( data_list = model_val_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_validation_data = True,
                                                                term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter, pad_output = False,
                                                                mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                ignore_output_errors = ignore_output_errors )

            # Encode Evaluation Data
            if len( model_eval_data ) > 0:
                self.Print_Log( "NERLink::Prepare_Model_Data() - Encoding CL Evaluation Data", force_print = True )
                eval_input, eval_output = self.Encode_Model_Data( data_list = model_eval_data, use_csr_format = self.model.Get_Use_CSR_Format(), is_evaluation_data = True,
                                                                  term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter, pad_output = False,
                                                                  mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                                  restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                  ignore_output_errors = ignore_output_errors )
        else:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Error: Model Type Not Supported" )
            return False

        # Check(s)
        if isinstance( train_input,  ( list, tuple, np.ndarray ) ) and len( train_input  ) == 0: train_input  = None
        if isinstance( train_output, ( list, tuple, np.ndarray ) ) and len( train_output ) == 0: train_output = None

        if data_file_path != "" and ( train_input is None or train_output is None ) or \
           eval_file_path != "" and ( eval_input is None or eval_output is None ):
            self.Print_Log( "NERLink::Prepare_Model_Data() - Error Occurred During Model Data Vectorization", force_print = True )
            return False

        # Force Skip Building Model i.e. We've Loaded A Pre-Trained Model For Inference/Fine-Tuning
        if skip_building_model:
            self.model_data_prepared = True
            return True

        # CSR Matrix Format
        if self.model.Get_Use_CSR_Format():
            input_shape  = train_input[0].shape  if isinstance( train_input,  tuple ) else train_input.shape
            output_shape = train_output[0].shape if isinstance( train_output, tuple ) else train_output.shape
            if train_input  is not None: self.Print_Log( "NERLink::Prepare_Model_Data() - Input Shape  : " + str( input_shape ) )
            if train_output is not None: self.Print_Log( "NERLink::Prepare_Model_Data() - Output Shape : " + str( output_shape ) )
            number_of_train_input_instances  = input_shape[0]
            number_of_train_output_instances = output_shape[0]
        # List/Array Format
        else:
            number_of_train_input_instances  = len( train_input )
            number_of_train_output_instances = len( train_output ) if train_output is not None else -1

            if train_input  is not None: self.Print_Log( "NERLink::Prepare_Model_Data() - Input Shape  : (" + str( len( train_input  ) ) + ", " + str( len( train_input[0]  ) ) + ")" )
            if train_output is not None: self.Print_Log( "NERLink::Prepare_Model_Data() - Output Shape : (" + str( len( train_output ) ) + ", " + str( len( train_output[0] ) ) + ")" )

        # Check(s)
        if number_of_train_input_instances == 0 or number_of_train_output_instances == 0:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Error Vectorizing Model Input/Output Data", force_print = True )
            return False

        if data_loader.Is_Embeddings_A_Loaded() and not data_loader.Simulate_Embeddings_A_Loaded_Mode():
            number_of_unique_terms = data_loader.Get_Number_Of_Embeddings_A()
        else:
            number_of_unique_terms = data_loader.Get_Number_Of_Unique_Tokens()

        # More Checks
        #   Check To See If Number Of Instances Is Divisible By Batch Size With No Remainder
        #   (Used For Model 'Batch_Generator' Function If 'use_csr_format == True')
        if self.model.Get_Use_CSR_Format() and number_of_train_input_instances % self.model.Get_Batch_Size() != 0:
            self.Print_Log( "NERLink::Prepare_Model_Data() - Warning: Number Of Instances Not Divisible By Batch Size" )
            self.Print_Log( "                              - Number Of Instances  : " + str( number_of_train_input_instances ) )
            self.Print_Log( "                              - Batch Size           : " + str( self.model.Get_Batch_Size()     ) )
            self.Print_Log( "                              - Batch_Generator Might Not Train Correctly / Change To Another Batch Size" )

            possible_batch_sizes = [ str( i ) if number_of_train_input_instances % i == 0 else "" for i in range( 1, number_of_train_input_instances ) ]
            possible_batch_sizes = " ".join( possible_batch_sizes )
            possible_batch_sizes = re.sub( r'\s+', ' ', possible_batch_sizes )

            self.Print_Log( "                              - Possible Batch Sizes : " + possible_batch_sizes )

        # Get Model Parameters From Data
        self.Print_Log( "NERLink::Prepare_Model_Data() - Fetching Model Parameters (Input/Output Sizes)" )
        if self.Get_Network_Model() == "concept_linking_embedding_similarity":
            number_of_inputs     = data_loader.Get_Embedding_A_Dimension()
            maximum_input_length = 1
            number_of_outputs    = data_loader.Get_Embedding_B_Dimension()
        elif self.Get_Network_Model() == "concept_linking_bert_embedding_similarity":
            number_of_inputs     = 1
            maximum_input_length = data_loader.Get_Max_Sequence_Length()
            number_of_outputs    = data_loader.Get_Embedding_B_Dimension()
        elif self.Get_Network_Model() == "ner_concept_linking_multi_task_bert":
            number_of_inputs     = 1
            maximum_input_length = data_loader.Get_Max_Sequence_Length()
            number_of_outputs    = ( data_loader.Get_Number_Of_Annotation_Labels(), data_loader.Get_Number_Of_Unique_Concepts() )
        else:
            number_of_inputs     = 1
            maximum_input_length = data_loader.Get_Max_Sequence_Length()
            number_of_outputs    = data_loader.Get_Number_Of_Annotation_Labels() if self.is_ner_model else data_loader.Get_Number_Of_Unique_Concepts()

        self.Print_Log( "                              - Number Of Unique Terms        : " + str( number_of_unique_terms   ) )
        self.Print_Log( "                              - Maximum Input Sequence Length : " + str( maximum_input_length     ) )
        self.Print_Log( "                              - Embedding Dimension Size      : " + str( embedding_dimension_size ) )
        self.Print_Log( "                              - Number Of Inputs              : " + str( number_of_inputs         ) )
        self.Print_Log( "                              - Number Of Outputs             : " + str( number_of_outputs        ) )

        self.model_data_prepared = True

        # Build Neural Network Model Based On Architecture
        self.Print_Log( "NERLink::Prepare_Model_Data() - Building Model" )
        self.Build_Model( maximum_sequence_length = maximum_input_length, number_of_unique_terms = number_of_unique_terms, embedding_dimension_size = embedding_dimension_size,
                          number_of_inputs = number_of_inputs, number_of_outputs = number_of_outputs, embeddings = data_loader.Get_Embeddings_A() )
        self.Print_Log( "NERLink::Prepare_Model_Data() - Complete" )
        return True

    # Checks If The NER or Concept Linking Model Has Been Built, If Not Builds The Model
    def Build_Model( self, maximum_sequence_length = 0, number_of_unique_terms = 0, embedding_dimension_size = 200, number_of_inputs = 0, number_of_outputs = 0, embeddings = [] ):
        # Check(s)
        if maximum_sequence_length == 0 or number_of_unique_terms == 0 or number_of_inputs == 0 or number_of_outputs == 0:
            self.Print_Log( "NERLink::Build_Model() - Error: One Or More Model Parameters == 0" )
            if maximum_sequence_length == 0: self.Print_Log( "NERLink::Build_Model() - Error: 'maximum_sequence_length == 0'", force_print = True )
            if number_of_unique_terms  == 0: self.Print_Log( "NERLink::Build_Model() - Error: 'number_of_unique_terms == 0'",  force_print = True )
            if number_of_inputs        == 0: self.Print_Log( "NERLink::Build_Model() - Error: 'number_of_inputs == 0'",        force_print = True )
            if number_of_outputs       == 0: self.Print_Log( "NERLink::Build_Model() - Error: 'number_of_outputs == 0'",       force_print = True )
            return False

        if len( embeddings ) == 0: self.Print_Log( "NERLink::Build_Model() - Warning: Number Of Embeddings == 0" )

        if not self.Is_Model_Loaded():
            if self.model.Get_Network_Model() in self.model_types:
                return self.model.Build_Model( max_sequence_length = maximum_sequence_length, number_of_unique_terms = number_of_unique_terms,
                                               embedding_dimension_size = embedding_dimension_size, number_of_inputs = number_of_inputs,
                                               number_of_outputs = number_of_outputs, embeddings = embeddings )
            else:
                self.Print_Log( "NERLink::Build_Model() - Error: Specified Network Model Not Supported", force_print = True )
                raise NotImplementedError
        else:
            self.Print_Log( "NERLink::Build_Model() - Warning: Model Has Already Been Built And Loaded In Memory" )

        return True

    """
        Updates Model Parameters
    """
    def Update_Model_Parameters( self, learning_rate = None, momentum = None, dropout = None, verbose = None, shuffle = None,
                                 per_epoch_saving = None, use_csr_format = None, embedding_a_path = None, embedding_b_path = None,
                                 trainable_weights = None, epochs = None, batch_size = None, margin = None, scale = None,
                                 learning_rate_decay = None, feature_scale_value = None, class_weights = None, sample_weights = None,
                                 use_cosine_annealing = None, cosine_annealing_min = None, cosine_annealing_max = None ):
        if learning_rate_decay is not None and learning_rate <= 0 or self.Get_Model().Get_Learning_Rate() <= 0:
            self.Print_Log( "NERLink::Update_Model_Parameters() - Error: Learning Rate Value Must Be >= 0", force_print = True )
            return False
        if feature_scale_value is not None and feature_scale_value <= 0 or self.Get_Model().Get_Feature_Scaling_Value() <= 0:
            self.Print_Log( "NERLink::Update_Model_Parameters() - Error: Feature Scaling Value Must Be >= 0", force_print = True )
            return False

        # NOTE: This Will Not Update Paramaters For Models Which Have Already Been Built
        # TODO: Update Model Parameters Which Have Already Been Built Using The Keras 'K.set_value()' Function.
        #       e.g. from keras import backend as K
        #            K.set_value(model.optimizer.learning_rate, 0.001)
        #            Source: https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        if learning_rate        is not None: self.model.Set_Learning_Rate( learning_rate )
        if momentum             is not None: self.model.Set_Momentum( momentum )
        if dropout              is not None: self.model.Set_Dropout( dropout )
        if verbose              is not None: self.model.Set_Verbose( verbose )
        if shuffle              is not None: self.model.Set_Shuffle( shuffle )
        if per_epoch_saving     is not None: self.model.Set_Per_Epoch_Saving( per_epoch_saving )
        if use_csr_format       is not None: self.model.Set_Use_CSR_Format( use_csr_format )
        if embedding_a_path     is not None: self.model.Set_Embedding_A_Path( embedding_a_path )
        if embedding_b_path     is not None: self.model.Set_Embedding_B_Path( embedding_b_path )
        if trainable_weights    is not None: self.model.Set_Trainable_Weights( trainable_weights )
        if epochs               is not None: self.model.Set_Epochs( epochs )
        if batch_size           is not None: self.model.Set_Batch_Size( batch_size )
        if margin               is not None: self.model.Set_Margin( margin )
        if scale                is not None: self.model.Set_Scale( scale )
        if learning_rate_decay  is not None: self.model.Set_Learning_Rate_Decay( learning_rate_decay )
        if feature_scale_value  is not None: self.model.Set_Feature_Scaling_Value( feature_scale_value )
        if class_weights        is not None: self.model.Set_Class_Weights( class_weights )
        if sample_weights       is not None: self.model.Set_Sample_Weights( sample_weights )
        if use_cosine_annealing is not None: self.model.Set_Use_Cosine_Annealing( use_cosine_annealing )
        if cosine_annealing_min is not None: self.model.Set_Cosine_Annealing_Min( cosine_annealing_min )
        if cosine_annealing_max is not None: self.model.Set_Cosine_Annealing_Max( cosine_annealing_max )

        return True

    """
       Trains NERLink Model
           Current Neural Architecture Implementations: BiLSTM, ELMo, BERT For NER & BERT+MLP, MLP, CosFace, ArcFace & SphereFace For Concept Linking
    """
    def Fit( self, training_file_path = "", validation_file_path = "", data_instances = [],
             encoded_input = [], encoded_output = [], val_encoded_input = [], val_encoded_output = [],
             embedding_a_path = None, embedding_b_path = None, learning_rate = None, learning_rate_decay = None,
             epochs = None, batch_size = None, shuffle = None, class_weights = None, sample_weights = None,
             use_cosine_annealing = None, cosine_annealing_min = None, cosine_annealing_max = None ):
        use_encoded_data, is_data_prepared, encoded_input_instances, encoded_output_instances = False, False, 0, 0

        # Fetch The Number Of Training Input Instances Depending On The Data Container Type
        if isinstance( encoded_input, list ): encoded_input = np.asarray( encoded_input )
        if isinstance( encoded_input, COO  ) or isinstance( encoded_input, np.ndarray ) or isinstance( encoded_input, csr_matrix ):
            encoded_input_instances = encoded_input.shape[0]
        elif isinstance( encoded_input, tuple ):
            encoded_input_instances = encoded_input[0].shape[0]     # BERT Models

        # Fetch The Number Of Training Output Instances Depending On The Data Container Type
        if isinstance( encoded_output, list ): encoded_output = np.asarray( encoded_output )
        if isinstance( encoded_output, COO  ) or isinstance( encoded_output, np.ndarray ) or isinstance( encoded_output, csr_matrix ):
            encoded_output_instances = encoded_output.shape[0]
        elif isinstance( encoded_output, tuple ):
            encoded_output_instances = encoded_output[0].shape[0]   # BERT Models

        # Check Data Format (Auto-Adjust CSR Format Setting)
        if encoded_input_instances > 0 and encoded_output_instances > 0:
            if isinstance( encoded_input, tuple ) or isinstance( encoded_output, tuple ):
                csr_format_flag = False

                # Check Inputs Within Tuple
                if isinstance( encoded_input, tuple ):
                    for input_element in encoded_input:
                        if isinstance( input_element, COO ) or isinstance( input_element, csr_matrix ):
                            csr_format_flag = True
                elif isinstance( encoded_input, COO ) or isinstance( encoded_input, csr_matrix ):
                    csr_format_flag = True

                # Check Outputs Within Tuple
                if isinstance( encoded_output, tuple ):
                    for output_element in encoded_output:
                        if isinstance( output_element, COO ) or isinstance( output_element, csr_matrix ):
                            csr_format_flag = True
                elif isinstance( encoded_output, COO ) or isinstance( encoded_output, csr_matrix ):
                    csr_format_flag = True

                # Set Use CSR Setting
                self.model.Set_Use_CSR_Format( csr_format_flag )
            else:
                if isinstance( encoded_input, COO  ) or isinstance( encoded_input, csr_matrix ) and not self.model.Get_Use_CSR_Format():
                    self.model.Set_Use_CSR_Format( True )
                elif isinstance( encoded_output, COO  ) or isinstance( encoded_output, csr_matrix ) and not self.model.Get_Use_CSR_Format():
                    self.model.Set_Use_CSR_Format( True )
                else:
                    self.model.Set_Use_CSR_Format( False )

        # Check(s)
        if encoded_input_instances == 0 and encoded_output_instances == 0:
            if training_file_path == "" and len( data_instances ) == 0:
                self.Print_Log( "NERLink::Fit() - Error: No Training File Path Specified Or Training Instance List Given", force_print = True )
                return False
            if self.utils.Check_If_File_Exists( training_file_path ) == False and len( data_instances ) == 0:
                self.Print_Log( "NERLink::Fit() - Error: Training File Data Path Does Not Exist", force_print = True )
                return False
        else:
            if encoded_input_instances == 0 or encoded_output_instances == 0:
                self.Print_Log( "NERLink::Fit() - Error: Encoded Inputs Or Outputs Contains Zero Instances " )
                return False

        # Update Model Parameters
        self.Update_Model_Parameters( learning_rate = learning_rate, learning_rate_decay = learning_rate_decay, class_weights = class_weights,
                                      epochs = epochs, batch_size = batch_size, shuffle = shuffle, embedding_a_path = embedding_a_path,
                                      embedding_b_path = embedding_b_path, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                                      cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )

        # Start Elapsed Time Timer
        start_time = time.time()

        # If Encoded/Vectorized Data Not Given, Prepare Model Data From Data File
        if encoded_input_instances == 0 and encoded_output_instances == 0:
            is_data_prepared = self.Prepare_Model_Data( data_file_path = training_file_path, val_file_path = validation_file_path,
                                                        data_instances = data_instances, embedding_dimension_size = self.model.Get_Number_Of_Hidden_Dimensions() )
        # We've Loaded Encoded/Vectorized Data
        elif not self.Is_Model_Loaded():
            number_of_inputs        = self.data_loader.Get_Number_Of_Embeddings_A() if self.data_loader.Is_Embeddings_A_Loaded() else self.data_loader.Get_Number_Of_Unique_Tokens()
            number_of_unique_tokens = self.data_loader.Get_Number_Of_Embeddings_A() if self.data_loader.Is_Embeddings_A_Loaded() else self.data_loader.Get_Number_Of_Unique_Tokens()
            number_of_outputs       = self.data_loader.Get_Number_Of_Annotation_Labels() if self.is_ner_model else self.data_loader.Get_Number_Of_Unique_Concepts()

            self.Build_Model( maximum_sequence_length = self.data_loader.Get_Max_Sequence_Length(), number_of_unique_terms = number_of_unique_tokens,
                              embedding_dimension_size = self.model.Get_Embedding_Dimension_Size(), number_of_inputs = number_of_inputs,
                              number_of_outputs = number_of_outputs, embeddings = self.data_loader.Get_Embeddings_A() )

            use_encoded_data, is_data_prepared = True, True
        elif encoded_input_instances > 0 and encoded_output_instances > 0:
            use_encoded_data, is_data_prepared = True, True

        # Check If Data Preparation Completed Successfully
        if is_data_prepared == False:
            self.Print_Log( "NERLink::Fit() - Error Preparing Data / Exiting Program", force_print = True )
            exit()

        # Check If Model Has Been Loaded/Created Prior To Continuing
        if not self.Is_Model_Loaded():
            self.Print_Log( "NERLink::Fit() - Error: Model Has Not Been Created/Loaded", force_print = True )
            return False

        #######################################################################
        #                                                                     #
        #   Execute Model Training                                            #
        #                                                                     #
        #######################################################################

        self.Print_Log( "NERLink::Fit() - Network Model Type - " + str( self.model.Get_Network_Model() ) )

        # Fetching Binarized Training Data From DataLoader Class
        self.Print_Log( "NERLink::Fit() - Fetching Model Inputs & Output Training Data" )

        if self.is_concept_linking_model:
            # Fetch Model Training Data
            if not use_encoded_data: encoded_input      = self.Get_Data_Loader().Get_Concept_Inputs()
            if not use_encoded_data: encoded_output     = self.Get_Data_Loader().Get_Concept_Outputs()

            # Fetch Model Validation Data
            if not use_encoded_data: val_encoded_input  = self.Get_Data_Loader().Get_Concept_Validation_Inputs()
            if not use_encoded_data: val_encoded_output = self.Get_Data_Loader().Get_Concept_Validation_Outputs()
        else:
            # Fetch Model Training Data
            if not use_encoded_data: encoded_input      = self.Get_Data_Loader().Get_NER_Inputs()
            if not use_encoded_data: encoded_output     = self.Get_Data_Loader().Get_NER_Outputs()

            # Fetch Model Validation Data
            if not use_encoded_data: val_encoded_input  = self.Get_Data_Loader().Get_NER_Validation_Inputs()
            if not use_encoded_data: val_encoded_output = self.Get_Data_Loader().Get_NER_Validation_Outputs()

        # Train Model
        self.model.Fit( encoded_input, encoded_output, val_encoded_input, val_encoded_output, epochs = self.model.Get_Epochs(),
                        batch_size = self.model.Get_Batch_Size(), verbose = self.model.Get_Verbose(), shuffle = self.model.Get_Shuffle(),
                        per_epoch_saving = self.model.Get_Per_Epoch_Saving(), class_weights = self.model.Get_Class_Weights(),
                        sample_weights = self.model.Get_Sample_Weights(), use_cosine_annealing = self.model.Get_Use_Cosine_Annealing(),
                        cosine_annealing_min = self.model.Get_Cosine_Annealing_Min(), cosine_annealing_max = self.model.Get_Cosine_Annealing_Max() )

        #######################################################################
        #                                                                     #
        #   Post Model Training                                               #
        #                                                                     #
        #######################################################################

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Fit() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        self.Print_Log( "NERLink::Fit() - Training Metrics:" )
        self.model.Print_Model_Training_Metrics()

        self.Print_Log( "NERLink::Fit() - Complete" )

        return True

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
          (NER)
            text_sequence        : Text Sequence To Encode (String)

          (CL)
            text_sequence        : Text Sequence In Which The 'entry_term' Occurs. (String)
            entry_term           : Concept Token (String)
            annotation_indices   : Concept Token Indices  (String Of Two Integers Separated By ':' Character)
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

          (Both)
            encoded_input        : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector        : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values    : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)
            instance_separator   : String Delimiter Used To Separate Model Data Instances (String)

        Outputs:
            prediction           : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, text_sequence = "", encoded_input = [], return_vector = False, return_raw_values = False, instance_separator = '<:>',
                 entry_term = "", annotation_indices = "", pad_input = True, pad_output = False, concept_delimiter = None,
                 mask_term_sequence = False, separate_sentences = True, term_sequence_only = False, restrict_context = False, label_per_sub_word = False ):
        # Check(s)
        if self.is_ner_model and len( encoded_input ) == 0 and text_sequence == "":
            self.Print_Log( "NERLink::Predict() - Error: NER Predictions Require Either A Raw Text Sequence Or Encoded Input Sequence", force_print = True )
            return []
        elif self.is_concept_linking_model and len( encoded_input ) == 0 and ( text_sequence == "" or entry_term == "" or annotation_indices == "" ):
            self.Print_Log( "NERLink::Predict() - Error: CL Predictions Require The Following Inputs: 'encoded_input' or 'text_sequence', 'entry_term' and 'annotation_indices'", force_print = True )
            return []

        if self.is_ner_model:
            return self.Predict_NER( text_sequence = text_sequence, encoded_input = encoded_input, return_vector = return_vector, return_raw_values = return_raw_values, instance_separator = instance_separator )
        elif self.is_concept_linking_model:
            return self.Predict_Concept( text_sequence = text_sequence, entry_term = entry_term, annotation_indices = annotation_indices, pad_input = pad_input, pad_output = pad_output,
                                         concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                         term_sequence_only = term_sequence_only, restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                         encoded_input = encoded_input, return_vector = return_vector, return_raw_values = return_raw_values, instance_separator = instance_separator )
        elif self.is_ner_cl_multi_task_model:
            return self.Predict_Multi_Task_NER_CL( text_sequence = text_sequence, entry_term = entry_term, annotation_indices = annotation_indices, pad_input = pad_input, pad_output = pad_output,
                                                   concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences,
                                                   term_sequence_only = term_sequence_only, restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                   encoded_input = encoded_input, return_vector = return_vector, return_raw_values = return_raw_values, instance_separator = instance_separator )

        return None

    """
        Outputs Model's NER Prediction Vector Given Inputs

        Inputs:
            text_sequence      : Text Sequence To Encode (String)
            encoded_input      : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector      : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values  : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)
            instance_separator : String Delimiter Used To Separate Model Data Instances (String)

        Outputs:
            prediction         : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict_NER( self, text_sequence = "", encoded_input = [], return_vector = False, return_raw_values = False, instance_separator = '<:>' ):
        # Checks(s)
        if text_sequence == "":    # We're Probably Just Using Input/Output Matrices
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Predict_NER() - Error: Input Matrix Is Empty" )
                return []

        # Start Elapsed Time Timer
        start_time = time.time()

        use_encoded_input, predictions = False, None

        if isinstance( encoded_input, list       ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, tuple      ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, COO        ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] > 0:
            use_encoded_input = True

        # Encode Model Input Data
        if not use_encoded_input:
            encoded_input = [self.Encode_NER_Instance( input_instance ) for input_instance in text_sequence.split( instance_separator )]
            encoded_input = np.asarray( encoded_input )

        # Predict Using Encoded Input Data
        predictions = self.model.Predict( encoded_input )

        # Check If Model Predicted Data
        if predictions is None:
            self.Print_Log( "NERLink::Predict_NER() - Error: Model Predictions Empty" )
            return []

        concept_token_to_idx = self.data_loader.Get_Annotation_Labels()
        annotation_idx_to_token = { v:k for k, v in concept_token_to_idx.items() }

        # Perform Thresholding On Raw Predition Vector/Matrix Values
        if return_vector:
            prediction_token_dim = predictions.shape[-1]
            for instance_index, prediction_instance in enumerate( predictions ):
                for token_index, prediction_element in enumerate( prediction_instance ):
                    predicted_annotation_index = np.argmax( prediction_element, axis = 0 )
                    predictions[instance_index][token_index] = np.zeros( ( prediction_token_dim ), dtype = np.float32 )
                    predictions[instance_index][token_index][predicted_annotation_index] = 1

        # Convert Predictions To Entity Label Tokens (List Of Strings: One For Each Token In A Given Instance)
        elif return_raw_values == False:
            # Store Labels For Each Prediction Instance
            label_predictions = []

            for prediction_instance in predictions:
                label_prediction_instance = []

                for prediction_element in prediction_instance:
                    predicted_annotation_index = np.argmax( prediction_element, axis = 0 )
                    label_prediction_instance.append( annotation_idx_to_token[predicted_annotation_index] )

                label_predictions.append( label_prediction_instance )

            predictions = label_predictions

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Predict_NER() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return predictions

    """
        Outputs Model's Concept Linking Prediction Vector Given Inputs

        Inputs:
            text_sequence        : Text Sequence In Which The 'entry_term' Occurs. (String)
            entry_term           : Concept Token (String)
            annotation_indices   : Concept Token Indices  (String Of Two Integers Separated By ':' Character)
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
            encoded_input        : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector        : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values    : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)
            instance_separator   : String Delimiter Used To Separate Model Data Instances (String)

        Outputs:
            prediction           : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict_Concept( self, text_sequence = "", encoded_input = [], return_vector = False, return_raw_values = False, instance_separator = '<:>',
                         entry_term = "", annotation_indices = "", pad_input = True, pad_output = False, concept_delimiter = None,
                         mask_term_sequence = False, separate_sentences = True, term_sequence_only = False, restrict_context = False, label_per_sub_word = False ):
        # Checks(s)
        if text_sequence == "":    # We're Probably Just Using Input/Output Matrices
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Predict_Concept() - Error: Input Matrix Is Empty" )
                return []

        # Start Elapsed Time Timer
        start_time = time.time()

        use_encoded_input, predictions = False, None

        if isinstance( encoded_input, list       ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, tuple      ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, COO        ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] > 0:
            use_encoded_input = True

        # Encode Model Input Data
        if not use_encoded_input:
            if "bert" in self.model.Get_Network_Model():
                self.Print_Log( "NERLink::Predict_Concept() - Encoding Token Using Text Sequence" )
                encoded_input, _ = self.Encode_CL_Instance( text_sequence = text_sequence, entry_term = entry_term,
                                                            annotation_concept = self.Get_Data_Loader().Get_CUI_LESS_Token(),
                                                            annotation_indices = annotation_indices, pad_input = pad_input, pad_output = pad_output,
                                                            concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                            separate_sentences = separate_sentences, term_sequence_only = term_sequence_only,
                                                            restrict_context = restrict_context, label_per_sub_word = label_per_sub_word )
                # If You Wanted To Peek Inside The Encoded Input (Debug Purposes)
                # input_ids, attention_masks, token_type_ids, entry_term_masks = encoded_input
                encoded_input = np.asarray( encoded_input, dtype = np.int32 )
            else:
                encoded_input = [[self.Get_Data_Loader().Get_Token_ID( input_instance )] for input_instance in text_sequence.split( instance_separator )]
                encoded_input = np.asarray( encoded_input, dtype = np.int32 )

        # Predict Using Encoded Input Data
        predictions = self.model.Predict( encoded_input )

        # Check If Model Predicted Data
        if predictions is None:
            self.Print_Log( "NERLink::Predict_Concept() - Error: Model Predictions Empty" )
            return []

        # Convert Predictions To Entity Label Tokens (List Of Strings: One For Each Token In A Given Instance)
        if return_vector == False:
            label_predictions = []

            concept_token_to_idx = self.data_loader.Get_Concept_ID_Dictionary()
            concept_idx_to_token = { v:k for k, v in concept_token_to_idx.items() }

            # Perform Thresholding Or ArgMax On Raw Predition Vector/Matrix Values
            #
            #   For BCE + Sigmoid:
            #       Rounding Of Raw Vector Values Per Instance Are Rounded Based On The Inflection Point Of The Sigmoid Function (0.5)
            #   For CCE + Softmax:
            #       ArgMax Is Taken On Prediction Instance
            for prediction_instance in predictions:
                # Either Threshold Prediction Instance Or Fetch Largest Value Of Probability Distribution
                #   i.e. 'binary_crossentropy' Loss Vs 'categorical_crossentropy' Loss
                if self.model.Get_Loss_Function() == "categorical_crossentropy":
                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[np.argmax( prediction_instance ).item()]] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[np.argmax( token_instance ).item()]] for token_instance in prediction_instance] )
                # 'sparse_categorical_crossentropy': Argmax Instance Output Vector And Set Max Index As Individual Array Containing Single Element
                elif self.model.Get_Loss_Function() == 'sparse_categorical_crossentropy':
                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[np.argmax( prediction_instance ).item()]] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[np.argmax( token_instance ).item()]] for token_instance in prediction_instance] )
                # 'binary_crossentropy'. Values > 0.5 Are Rounded To 1 While Values <= 0.5 Are Reduced To 0
                elif self.model.Get_Loss_Function() == 'binary_crossentropy':
                    prediction_instance = np.round( prediction_instance )

                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[idx] for idx, value in enumerate( prediction_instance ) if value == 1.0] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[idx] for idx, value in enumerate( token_labels ) if value == 1.0] for token_labels in prediction_instance] )
                else:
                    label_predictions.append( prediction_instance )

            predictions = label_predictions

        # Perform Thresholding Or ArgMax On Raw Predition Vector/Matrix Value
        #   For BCE + Sigmoid:
        #       Rounding Of Raw Vector Values Per Instance Are Rounded Based On The Inflection Point Of The Sigmoid Function (0.5)
        #   For CCE + Softmax:
        #       ArgMax Is Taken On Prediction Instance
        elif return_raw_values == False:
            label_predictions = []

            for instance_index, prediction_instance in enumerate( predictions ):
                # Either Threshold Prediction Instance Or Fetch Largest Value Of Probability Distribution
                #   i.e. 'binary_crossentropy' Loss vs 'categorical_crossentropy' vs 'sparse_categorical_crossentropy' Loss
                if self.model.Get_Loss_Function() == "categorical_crossentropy":
                    if prediction_instance.ndim == 1:
                        label_predictions.append( np.asarray( [ 1 if idx == np.argmax( prediction_instance ).item() else 0 for idx, val in enumerate( prediction_instance ) ], dtype = np.int32 ) )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( np.asarray( [ [1 if idx == np.argmax( token_instance ).item() else 0 for idx, _ in enumerate( token_instance )] for token_instance in prediction_instance ], dtype = np.int32 ) )
                # 'sparse_categorical_crossentropy': Argmax Instance Output Vector And Set Max Index As Individual Array Containing Single Element
                elif self.model.Get_Loss_Function() == 'sparse_categorical_crossentropy':
                    if prediction_instance.ndim == 1:
                        label_predictions.append( np.asarray( [ 1 if idx == np.argmax( prediction_instance ).item() else 0 for idx, val in enumerate( prediction_instance ) ], dtype = np.int32 ) )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( np.asarray( [ [1 if idx == np.argmax( token_instance ).item() else 0 for idx, _ in enumerate( token_instance )] for token_instance in prediction_instance ], dtype = np.int32 ) )
                # 'binary_crossentropy': Values > 0.5 Are Rounded To 1 While Values <= 0.5 Are Reduced To 0
                elif self.model.Get_Loss_Function() == "binary_crossentropy":
                    label_predictions.append( np.round( prediction_instance ) )

            predictions = np.asarray( label_predictions, dtype = np.int32 )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Predict_Concept() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return predictions

    """
        Outputs Model's NER + Concept Linking Prediction Vector Given Inputs

        Inputs:
            text_sequence        : Text Sequence In Which The 'entry_term' Occurs. (String)
            entry_term           : Concept Token (String)
            annotation_indices   : Concept Token Indices  (String Of Two Integers Separated By ':' Character)
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
            encoded_input        : Model Input Matrix Of One Or More Vectorized Model Input (ie. Output From DataLoader::Vectorize_Model_Data() / DataLoader::Vectorize_Model_Inputs() functions).
            return_vector        : True = Return Prediction Vector, False = Return Predicted Tokens (Boolean)
            return_raw_values    : True = Output Raw Prediction Values From Model / False = Output Values After Passing Through Prediction Threshold (Boolean)
            instance_separator   : String Delimiter Used To Separate Model Data Instances (String)

        Outputs:
            prediction           : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict_Multi_Task_NER_CL( self, text_sequence = "", encoded_input = [], return_vector = False, return_raw_values = False, instance_separator = '<:>',
                                   entry_term = "", annotation_indices = "", pad_input = True, pad_output = False, concept_delimiter = None,
                                   mask_term_sequence = False, separate_sentences = True, term_sequence_only = False, restrict_context = False, label_per_sub_word = False ):
        # Checks(s)
        if text_sequence == "":    # We're Probably Just Using Input/Output Matrices
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Error: Input Matrix Is Empty" )
                return []

        # Start Elapsed Time Timer
        start_time = time.time()

        use_encoded_input, predictions = False, None

        if isinstance( encoded_input, list       ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, tuple      ) and len( encoded_input )   > 0 or \
           isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, COO        ) and encoded_input.shape[0] > 0 or \
           isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] > 0:
            use_encoded_input = True

        # Encode Model Input Data
        if not use_encoded_input:
            self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Encoding Token Using Text Sequence" )
            encoded_input, _ = self.Encode_CL_Instance( text_sequence = text_sequence, entry_term = entry_term,
                                                        annotation_concept = self.Get_Data_Loader().Get_CUI_LESS_Token(),
                                                        annotation_indices = annotation_indices, pad_input = pad_input, pad_output = pad_output,
                                                        concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                        separate_sentences = separate_sentences, term_sequence_only = term_sequence_only,
                                                        restrict_context = restrict_context, label_per_sub_word = label_per_sub_word )
            # If You Wanted To Peek Inside The Encoded Input (Debug Purposes)
            # input_ids, attention_masks, token_type_ids, entry_term_masks = encoded_input
            token_ids, attention_mask, token_type_ids, entry_term_masks = encoded_input
            encoded_input = ( token_ids, attention_mask, token_type_ids, entry_term_masks )
            encoded_input = np.asarray( encoded_input, dtype = np.int32 )

        # Predict Using Encoded Input Data
        predictions = self.model.Predict( encoded_input )

        # Check(s)
        if len( predictions ) < 2:
            self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Error: Model Prediction Length < 2 Elements" )
            return []

        ner_predictions, cl_predictions     = predictions
        ner_loss_function, cl_loss_function = None, None

        if isinstance( self.model.Get_Loss_Function(), dict ):
            ner_loss_function, cl_loss_function = self.model.Get_Loss_Function().values()
        else:
            ner_loss_function, cl_loss_function = "sparse_categorical_crossentropy", "binary_crossentropy"

        # --------------------------- #
        #       NER Predictions       #
        # --------------------------- #

        # Check If Model Predicted Data
        if ner_predictions is None or len( ner_predictions ) == 0:
            self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Error: Model NER Predictions Empty" )
            return []

        concept_token_to_idx = self.data_loader.Get_Annotation_Labels()
        annotation_idx_to_token = { v:k for k, v in concept_token_to_idx.items() }

        # Convert Predictions To Entity Label Tokens (List Of Strings: One For Each Token In A Given Instance)
        if return_vector == False:
            # Store Labels For Each Prediction Instance
            label_predictions = []

            for prediction_instance in ner_predictions:
                label_prediction_instance = []

                for prediction_element in prediction_instance:
                    predicted_annotation_index = np.argmax( prediction_element, axis = 0 )
                    label_prediction_instance.append( annotation_idx_to_token[predicted_annotation_index] )

                label_predictions.append( label_prediction_instance )

            ner_predictions = label_predictions

        # Perform Thresholding On Raw Predition Vector/Matrix Values
        elif return_raw_values == False:
            prediction_token_dim = ner_predictions.shape[-1]
            for instance_index, prediction_instance in enumerate( ner_predictions ):
                for token_index, prediction_element in enumerate( prediction_instance ):
                    predicted_annotation_index = np.argmax( prediction_element, axis = 0 )
                    ner_predictions[instance_index][token_index] = np.zeros( ( prediction_token_dim ), dtype = np.float32 )
                    ner_predictions[instance_index][token_index][predicted_annotation_index] = 1

        # --------------------------------------- #
        #       Concept Linking Predictions       #
        # --------------------------------------- #

        # Check Model Predicted Data
        if cl_predictions is None or len( cl_predictions ) == 0:
            self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Error: Model CL Predictions Empty" )
            return []

        # Convert Predictions To Entity Label Tokens (List Of Strings: One For Each Token In A Given Instance)
        if return_vector == False:
            label_predictions = []

            concept_token_to_idx = self.data_loader.Get_Concept_ID_Dictionary()
            concept_idx_to_token = { v:k for k, v in concept_token_to_idx.items() }

            # Perform Thresholding Or ArgMax On Raw Predition Vector/Matrix Values
            #
            #   For BCE + Sigmoid:
            #       Rounding Of Raw Vector Values Per Instance Are Rounded Based On The Inflection Point Of The Sigmoid Function (0.5)
            #   For CCE + Softmax:
            #       ArgMax Is Taken On Prediction Instance
            for prediction_instance in cl_predictions:
                # Either Threshold Prediction Instance Or Fetch Largest Value Of Probability Distribution
                #   i.e. 'binary_crossentropy' Loss Vs 'categorical_crossentropy' Loss
                if cl_loss_function == "categorical_crossentropy":
                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[np.argmax( prediction_instance ).item()]] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[np.argmax( token_instance ).item()]] for token_instance in prediction_instance] )
                # 'sparse_categorical_crossentropy': Argmax Instance Output Vector And Set Max Index As Individual Array Containing Single Element
                elif cl_loss_function == 'sparse_categorical_crossentropy':
                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[np.argmax( prediction_instance ).item()]] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[np.argmax( token_instance ).item()]] for token_instance in prediction_instance] )
                # 'binary_crossentropy'. Values > 0.5 Are Rounded To 1 While Values <= 0.5 Are Reduced To 0
                elif cl_loss_function == 'binary_crossentropy':
                    prediction_instance = np.round( prediction_instance )

                    if prediction_instance.ndim == 1:
                        label_predictions.append( [concept_idx_to_token[idx] for idx, value in enumerate( prediction_instance ) if value == 1.0] )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( [[concept_idx_to_token[idx] for idx, value in enumerate( token_labels ) if value == 1.0] for token_labels in prediction_instance] )
                else:
                    label_predictions.append( prediction_instance )

            cl_predictions = label_predictions

        # Perform Thresholding Or ArgMax On Raw Predition Vector/Matrix Value
        #   For BCE + Sigmoid:
        #       Rounding Of Raw Vector Values Per Instance Are Rounded Based On The Inflection Point Of The Sigmoid Function (0.5)
        #   For CCE + Softmax:
        #       ArgMax Is Taken On Prediction Instance
        elif return_raw_values == False:
            label_predictions = []

            for instance_index, prediction_instance in enumerate( cl_predictions ):
                # Either Threshold Prediction Instance Or Fetch Largest Value Of Probability Distribution
                #   i.e. 'binary_crossentropy' Loss vs 'categorical_crossentropy' vs 'sparse_categorical_crossentropy' Loss
                if cl_loss_function == "categorical_crossentropy":
                    if prediction_instance.ndim == 1:
                        label_predictions.append( np.asarray( [ 1 if idx == np.argmax( prediction_instance ).item() else 0 for idx, val in enumerate( prediction_instance ) ], dtype = np.int32 ) )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( np.asarray( [ [1 if idx == np.argmax( token_instance ).item() else 0 for idx, _ in enumerate( token_instance )] for token_instance in prediction_instance ], dtype = np.int32 ) )
                # 'sparse_categorical_crossentropy': Argmax Instance Output Vector And Set Max Index As Individual Array Containing Single Element
                elif cl_loss_function == 'sparse_categorical_crossentropy':
                    if prediction_instance.ndim == 1:
                        label_predictions.append( np.asarray( [ 1 if idx == np.argmax( prediction_instance ).item() else 0 for idx, val in enumerate( prediction_instance ) ], dtype = np.int32 ) )
                    elif prediction_instance.ndim == 2:
                        label_predictions.append( np.asarray( [ [1 if idx == np.argmax( token_instance ).item() else 0 for idx, _ in enumerate( token_instance )] for token_instance in prediction_instance ], dtype = np.int32 ) )
                # 'binary_crossentropy': Values > 0.5 Are Rounded To 1 While Values <= 0.5 Are Reduced To 0
                elif cl_loss_function == "binary_crossentropy":
                    label_predictions.append( np.round( prediction_instance ) )

            cl_predictions = np.asarray( label_predictions, dtype = np.int32 )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Predict_Multi_Task_NER_CL() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return ( ner_predictions, cl_predictions )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, test_file_path = "", encoded_input = [], encoded_output = [], lowercase = True ):
        load_from_file = True

        # Check(s)
        if test_file_path != "":
            if self.Is_Model_Loaded() == False:
                self.Print_Log( "NERLink::Evaluate() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
                return -1
            if self.utils.Check_If_File_Exists( test_file_path ) == False:
                self.Print_Log( "NERLink::Evaluate() - Error: Data File Does Not Exist At Path", force_print = True )
                return -1
        else:
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate() - Error: Input Matrix Is Empty", force_print = True )
                return -1
            if isinstance( encoded_output, list       ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, tuple      ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, np.ndarray ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, COO        ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, csr_matrix ) and encoded_output.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate() - Error: Output Matrix Is Empty", force_print = True )
                return -1

            # Bug Fix - Fixes Potential User 'encoded_input' And 'encoded_output' Error.
            #           For A Single Instance Contains 'self.Get_Data_Loader().Get_Max_Sequence_Length()' Elements
            #           This Adds The Extra Dimension To Conform To Evaluation Standards.
            #           Single Instance For NER Output Is 2 Dimensions, More Than One Instance = 3 Dimensions
            if isinstance( encoded_input, np.ndarray ) and encoded_input.ndim  == 1: encoded_input  = np.expand_dims( encoded_input, axis = 0 )

            if self.is_ner_model:
                if not isinstance( encoded_input, tuple ) and isinstance( encoded_output, np.ndarray ) and encoded_output.ndim == 2:
                    encoded_output = np.expand_dims( encoded_output, axis = 0 )
            elif self.is_concept_linking_model:
                if not isinstance( encoded_input, tuple ) and isinstance( encoded_output, np.ndarray ) and encoded_output.ndim == 1:
                    encoded_output = np.expand_dims( encoded_output, axis = 0 )

            number_of_input_instances, number_of_output_instances = 0, 0

            # Fetch The Number Of Input Instances Depending On The Data Container Type
            if   isinstance( encoded_input, list  ): number_of_input_instances = len( encoded_input )
            elif isinstance( encoded_input, COO   ) or isinstance( encoded_input, csr_matrix ) or isinstance( encoded_input, np.ndarray ):
                number_of_input_instances = encoded_input.shape[0]
            elif isinstance( encoded_input, tuple ): number_of_input_instances = encoded_input[0].shape[0]     # BERTModel

            # Fetch The Number Of Output Instances Depending On The Data Container Type
            if   isinstance( encoded_output, list  ): number_of_output_instances = len( encoded_output )
            elif isinstance( encoded_output, COO   ) or isinstance( encoded_output, csr_matrix ) or isinstance( encoded_output, np.ndarray ):
                number_of_output_instances = encoded_output.shape[0]
            elif isinstance( encoded_output, tuple ): number_of_output_instances = encoded_output[0].shape[0]

            # Determine If We Have An Input-to-Output Number Of Instance Mismatch
            if number_of_input_instances != number_of_output_instances:
                self.Print_Log( "NERLink::Evaluate() - Error: Number Of Input Instances != Number Of Output Instances", force_print = True )
                self.Print_Log( "NERLink::Evaluate() -        Number Of Input Instances : " + str( number_of_input_instances  ), force_print = True )
                self.Print_Log( "NERLink::Evaluate() -        Number Of Output Instances: " + str( number_of_output_instances ), force_print = True )
                return -1

            load_from_file = False

        # Start Elapsed Time Timer
        start_time = time.time()

        # Load Data From File And Perform Data Preprocessing Steps
        if load_from_file:
            data_loader = self.Get_Data_Loader()
            data_list = data_loader.Read_Data( test_file_path, keep_in_memory = False, lowercase = lowercase )

            if len( data_list ) == 0:
                self.Print_Log( "NERLink::Evaluate_Manual() - Error: Test Data List Is Empty" )
                return -1

            if self.is_ner_model:
                encoded_input, encoded_output = data_loader.Encode_NER_Model_Data( data_list, use_csr_format = self.Get_Model().Get_Use_CSR_Format() )
            elif self.is_concept_linking_model:
                encoded_input, encoded_output = data_loader.Encode_CL_Model_Data( data_list, use_csr_format = self.Get_Model().Get_Use_CSR_Format() )

        # Convert 'COO' and 'csr_matrix' To Dense Format
        if   isinstance( encoded_input, COO ):         encoded_input  = encoded_input.todense()
        elif isinstance( encoded_input, csr_matrix ):  encoded_input  = encoded_input.todense()
        elif isinstance( encoded_input, tuple ):
            encoded_input = list( encoded_input )

            for idx in range( len( encoded_input ) ):
                if isinstance( encoded_input[idx], COO ) or isinstance( encoded_input[idx], csr_matrix ):
                    encoded_input[idx] = encoded_input[idx].todense()

            encoded_input = tuple( encoded_input )

        if   isinstance( encoded_output, COO ):        encoded_output = encoded_output.todense()
        elif isinstance( encoded_output, csr_matrix ): encoded_output = encoded_output.todense()
        elif isinstance( encoded_output, tuple ):
            encoded_output = list( encoded_output )

            for idx in range( len( encoded_output ) ):
                if isinstance( encoded_output[idx], COO ) or isinstance( encoded_output[idx], csr_matrix ):
                    encoded_output[idx] = encoded_output[idx].todense()

            encoded_output = tuple( encoded_output )

        loss, accuracy, precision, recall, f1_score = self.model.Evaluate( encoded_input, encoded_output, verbose = self.Get_Model().Get_Verbose() )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Evaluate() - Elapsed Time: " + str( elapsed_time ) + " secs", force_print = True )

        return loss, accuracy, precision, recall, f1_score

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)
            encoded_input  : Encoded Data Input Matrix
            encoded_output : Encoded Data Output Matrix / Ground Truth Labels

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_Manual( self, test_file_path = "", encoded_input = [], encoded_output = [], lowercase = True ):
        if self.is_ner_model:
            return self.Evaluate_NER( test_file_path = test_file_path, encoded_input = encoded_input, encoded_output = encoded_output, lowercase = lowercase )
        elif self.is_concept_linking_model:
            return self.Evaluate_Concept_Linking( test_file_path = test_file_path, encoded_input = encoded_input, encoded_output = encoded_output, lowercase = lowercase )
        elif self.is_ner_cl_multi_task_model:
            return self.Evaluate_NER_Concept_Linking( test_file_path = test_file_path, encoded_input = encoded_input, encoded_output = encoded_output, lowercase = lowercase )
        return None

    """
        Evaluates Model's Ability To Predict NER Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)
            encoded_input  : Encoded Data Input Matrix
            encoded_output : Encoded Data Output Matrix / Ground Truth Labels

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_NER( self, test_file_path = "", encoded_input = [], encoded_output = [], lowercase = True ):
        load_from_file = True

        # Check(s)
        if test_file_path != "":
            if self.Is_Model_Loaded() == False:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
                return -1
            if self.utils.Check_If_File_Exists( test_file_path ) == False:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: Data File Does Not Exist At Path", force_print = True )
                return -1
        else:
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: Input Matrix Is Empty", force_print = True )
                return -1
            if isinstance( encoded_output, list       ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, tuple      ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, np.ndarray ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, COO        ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, csr_matrix ) and encoded_output.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: Output Matrix Is Empty", force_print = True )
                return -1

            # Bug Fix - Fixes Potential User 'encoded_input' And 'encoded_output' Error.
            #           For A Single Instance Contains 'self.Get_Data_Loader().Get_Max_Sequence_Length()' Elements
            #           This Adds The Extra Dimension To Conform To Evaluation Standards.
            #           Single Instance For NER Output Is 2 Dimensions, More Than One Instance = 3 Dimensions
            #           (Ignore If 'encoded_input' Is Tuple i.e. BERTModel Input/Output)
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_input,  np.ndarray ) and encoded_input.ndim  == 1:
                encoded_input  = np.expand_dims( encoded_input,  axis = 0 )
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_output, np.ndarray ) and encoded_output.ndim == 2:
                encoded_output = np.expand_dims( encoded_output, axis = 0 )

            number_of_input_instances, number_of_output_instances = 0, 0

            # Fetch The Number Of Input Instances Depending On The Data Container Type
            if   isinstance( encoded_input, list  ): number_of_input_instances = len( encoded_input )
            elif isinstance( encoded_input, COO   ) or isinstance( encoded_input, csr_matrix ) or isinstance( encoded_input, np.ndarray ):
                number_of_input_instances = encoded_input.shape[0]
            elif isinstance( encoded_input, tuple ): number_of_input_instances = encoded_input[0].shape[0]     # BERTModel

            # Fetch The Number Of Output Instances Depending On The Data Container Type
            if   isinstance( encoded_output, list  ): number_of_output_instances = len( encoded_output )
            elif isinstance( encoded_output, COO   ) or isinstance( encoded_output, csr_matrix ) or isinstance( encoded_output, np.ndarray ):
                number_of_output_instances = encoded_output.shape[0]
            elif isinstance( encoded_output, tuple ): number_of_output_instances = encoded_output[0].shape[0]

            # Determine If We Have An Input-to-Output Number Of Instance Mismatch
            if number_of_input_instances != number_of_output_instances:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: Number Of Input Instances != Number Of Output Instances", force_print = True )
                self.Print_Log( "NERLink::Evaluate_NER() -        Number Of Input Instances : " + str( number_of_input_instances  ), force_print = True )
                self.Print_Log( "NERLink::Evaluate_NER() -        Number Of Output Instances: " + str( number_of_output_instances ), force_print = True )
                return -1

            load_from_file = False

        # Start Elapsed Time Timer
        start_time = time.time()

        # Load Data From File And Perform Data Preprocessing Steps
        if load_from_file:
            data_loader = self.Get_Data_Loader()
            data_list = data_loader.Read_Data( test_file_path, keep_in_memory = False, lowercase = lowercase )

            if len( data_list ) == 0:
                self.Print_Log( "NERLink::Evaluate_NER() - Error: Test Data List Is Empty" )
                return -1

            encoded_input, encoded_output = data_loader.Encode_NER_Model_Data( data_list, use_csr_format = self.Get_Model().Get_Use_CSR_Format() )

            # Fetch ELMo Input If Model Architecutre Is "ELMo"
            if self.model.Get_Network_Model() == "ner_elmo":
                encoded_input = data_loader.Tokenize_Model_Data( data_list = data_list )
                encoded_input = np.asarray( encoded_input, object )

        # Perform NER Inference Over Inputs
        model_predictions = self.Predict_NER( encoded_input = encoded_input, return_vector = True, return_raw_values = False )

        # Convert Matrix To Dense Format
        if isinstance( encoded_input,  COO ) or isinstance( encoded_input,  csr_matrix ): encoded_input  = encoded_input.todense()
        if isinstance( encoded_output, COO ) or isinstance( encoded_output, csr_matrix ): encoded_output = encoded_output.todense()
        if isinstance( encoded_input,  np.matrixlib.matrix ): encoded_input  = np.asarray( encoded_input  )
        if isinstance( encoded_output, np.matrixlib.matrix ): encoded_output = np.asarray( encoded_output )

        # Count Number Of Elements Which Are Not Padding In Input Instances
        if self.model.Get_Network_Model() == "ner_elmo":
            if isinstance( encoded_input, list ): encoded_input = np.asarray( encoded_input, object )
            actual_instance_sizes = np.count_nonzero( encoded_input != "", axis = -1, keepdims = True )
        elif self.model.Get_Network_Model() == "ner_bert":
            padding_token = self.Get_Data_Loader().Get_Padding_Token()
            padding_id    = self.Get_Data_Loader().Get_Token_ID_Dictionary()[padding_token]
            actual_instance_sizes = np.count_nonzero( encoded_input[0] != padding_id, axis = -1, keepdims = True )
        else:
            padding_token = self.Get_Data_Loader().Get_Padding_Token()
            padding_id    = self.Get_Data_Loader().Get_Token_ID( padding_token )
            actual_instance_sizes = np.count_nonzero( encoded_input != padding_id, axis = -1, keepdims = True )

        metrics = self.Generate_Classification_Report( model_predictions, encoded_output, actual_instance_sizes )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Evaluate_NER() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return metrics

    """
        Evaluates Model's Ability To Predict Concept Linking Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)
            encoded_input  : Encoded Data Input Matrix
            encoded_output : Encoded Data Output Matrix / Ground Truth Labels

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_Concept_Linking( self, test_file_path = "", encoded_input = [], encoded_output = [], lowercase = True ):
        load_from_file = True

        # Check(s)
        if test_file_path != "":
            if self.Is_Model_Loaded() == False:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
                return -1
            if self.utils.Check_If_File_Exists( test_file_path ) == False:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: Data File Does Not Exist At Path", force_print = True )
                return -1
        else:
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: Input Matrix Is Empty", force_print = True )
                return -1
            if isinstance( encoded_output, list       ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, tuple      ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, np.ndarray ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, COO        ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, csr_matrix ) and encoded_output.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: Output Matrix Is Empty", force_print = True )
                return -1

            # Bug Fix - Fixes Potential User 'encoded_input' And 'encoded_output' Error.
            #           For A Single Instance Contains 'self.Get_Data_Loader().Get_Max_Sequence_Length()' Elements
            #           This Adds The Extra Dimension To Conform To Evaluation Standards.
            #           Single Instance For NER Output Is 2 Dimensions, More Than One Instance = 3 Dimensions
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_input,  np.ndarray ) and encoded_input.ndim  == 1:
                encoded_input  = np.expand_dims( encoded_input,  axis = 0 )
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_output, np.ndarray ) and encoded_output.ndim == 1:
                encoded_output = np.expand_dims( encoded_output, axis = 0 )

            number_of_input_instances, number_of_output_instances = 0, 0

            # Fetch The Number Of Input Instances Depending On The Data Container Type
            if   isinstance( encoded_input, list ): number_of_input_instances = len( encoded_input )
            elif isinstance( encoded_input, COO  ) or isinstance( encoded_input, csr_matrix ) or isinstance( encoded_input, np.ndarray ):
                number_of_input_instances = encoded_input.shape[0]
            elif isinstance( encoded_input, tuple ): number_of_input_instances = encoded_input[0].shape[0]     # BERTModel

            # Fetch The Number Of Output Instances Depending On The Data Container Type
            if   isinstance( encoded_output, list ): number_of_output_instances = len( encoded_output )
            elif isinstance( encoded_output, COO  ) or isinstance( encoded_output, csr_matrix ) or isinstance( encoded_output, np.ndarray ):
                number_of_output_instances = encoded_output.shape[0]
            elif isinstance( encoded_output, tuple ): number_of_output_instances = encoded_output[0].shape[0]

            # Determine If We Have An Input-to-Output Number Of Instance Mismatch
            if number_of_input_instances != number_of_output_instances:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: Number Of Input Instances != Number Of Output Instances", force_print = True )
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() -        Number Of Input Instances : " + str( number_of_input_instances  ), force_print = True )
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() -        Number Of Output Instances: " + str( number_of_output_instances ), force_print = True )
                return -1

            load_from_file = False

        # Start Elapsed Time Timer
        start_time = time.time()

        # Load Data From File And Perform Data Preprocessing Steps
        if load_from_file:
            data_loader = self.Get_Data_Loader()
            data_list   = data_loader.Read_Data( test_file_path, keep_in_memory = False, lowercase = lowercase )

            if len( data_list ) == 0:
                self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Error: Test Data List Is Empty" )
                return -1

            encoded_input, encoded_output = data_loader.Encode_CL_Model_Data( data_list, use_csr_format = self.Get_Model().Get_Use_CSR_Format() )

        # Perform Concept Linking Inference Over Inputs
        model_predictions = self.Predict_Concept( encoded_input = encoded_input, return_vector = True, return_raw_values = False )

        # Convert Matrix To Dense Format
        if isinstance( encoded_input,  COO ) or isinstance( encoded_input,  csr_matrix ): encoded_input  = encoded_input.todense()
        if isinstance( encoded_output, COO ) or isinstance( encoded_output, csr_matrix ): encoded_output = encoded_output.todense()
        if isinstance( encoded_input,  np.matrixlib.matrix ): encoded_input  = np.asarray( encoded_input  )
        if isinstance( encoded_output, np.matrixlib.matrix ): encoded_output = np.asarray( encoded_output )

        # Compute Actual Instance Sub-Word Lengths
        actual_instance_sizes = []

        if len( actual_instance_sizes ) == 0 and "bert" in self.model.Get_Network_Model():
            # Use Input Sub-Word IDs To Determine Actual Lengths
            for sub_word_instance_input_ids in encoded_input[0]:
                non_padding_sub_words = [ sub_word_id for sub_word_id in sub_word_instance_input_ids if sub_word_id != self.Get_Data_Loader().Get_Sub_Word_PAD_Token_ID() ]
                actual_instance_sizes.append( len( non_padding_sub_words ) )

        metrics = self.Generate_Classification_Report( model_predictions = model_predictions, true_labels = encoded_output, actual_instance_sizes = actual_instance_sizes )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Evaluate_Concept_Linking() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return metrics

    """
        Evaluates Model's Ability To Predict NER & Concept Linking Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)
            encoded_input  : Encoded Data Input Matrix
            encoded_output : Encoded Data Output Matrix / Ground Truth Labels

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate_NER_Concept_Linking( self, test_file_path = "", encoded_input = [], encoded_output = [], lowercase = True ):
        load_from_file = True

        # Check(s)
        if test_file_path != "":
            if self.Is_Model_Loaded() == False:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
                return -1
            if self.utils.Check_If_File_Exists( test_file_path ) == False:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: Data File Does Not Exist At Path", force_print = True )
                return -1
        else:
            if isinstance( encoded_input, list       ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, tuple      ) and len( encoded_input )   == 0 or \
               isinstance( encoded_input, np.ndarray ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, COO        ) and encoded_input.shape[0] == 0 or \
               isinstance( encoded_input, csr_matrix ) and encoded_input.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: Input Matrix Is Empty", force_print = True )
                return -1
            if isinstance( encoded_output, list       ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, tuple      ) and len( encoded_output )   == 0 or \
               isinstance( encoded_output, np.ndarray ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, COO        ) and encoded_output.shape[0] == 0 or \
               isinstance( encoded_output, csr_matrix ) and encoded_output.shape[0] == 0:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: Output Matrix Is Empty", force_print = True )
                return -1

            # Bug Fix - Fixes Potential User 'encoded_input' And 'encoded_output' Error.
            #           For A Single Instance Contains 'self.Get_Data_Loader().Get_Max_Sequence_Length()' Elements
            #           This Adds The Extra Dimension To Conform To Evaluation Standards.
            #           Single Instance For NER Output Is 2 Dimensions, More Than One Instance = 3 Dimensions
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_input,  np.ndarray ) and encoded_input.ndim  == 1:
                encoded_input  = np.expand_dims( encoded_input,  axis = 0 )
            if not isinstance( encoded_input, tuple ) and isinstance( encoded_output, np.ndarray ) and encoded_output.ndim == 1:
                encoded_output = np.expand_dims( encoded_output, axis = 0 )

            number_of_input_instances, number_of_output_instances = 0, 0

            # Fetch The Number Of Input Instances Depending On The Data Container Type
            if   isinstance( encoded_input, list ): number_of_input_instances = len( encoded_input )
            elif isinstance( encoded_input, COO  ) or isinstance( encoded_input, csr_matrix ) or isinstance( encoded_input, np.ndarray ):
                number_of_input_instances = encoded_input.shape[0]
            elif isinstance( encoded_input, tuple ): number_of_input_instances = encoded_input[0].shape[0]     # BERTModel

            # Fetch The Number Of Output Instances Depending On The Data Container Type
            if   isinstance( encoded_output, list ): number_of_output_instances = len( encoded_output )
            elif isinstance( encoded_output, COO  ) or isinstance( encoded_output, csr_matrix ) or isinstance( encoded_output, np.ndarray ):
                number_of_output_instances = encoded_output.shape[0]
            elif isinstance( encoded_output, tuple ): number_of_output_instances = encoded_output[0].shape[0]

            # Determine If We Have An Input-to-Output Number Of Instance Mismatch
            if number_of_input_instances != number_of_output_instances:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: Number Of Input Instances != Number Of Output Instances", force_print = True )
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() -        Number Of Input Instances : " + str( number_of_input_instances  ), force_print = True )
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() -        Number Of Output Instances: " + str( number_of_output_instances ), force_print = True )
                return -1

            load_from_file = False

        # Start Elapsed Time Timer
        start_time = time.time()

        # Load Data From File And Perform Data Preprocessing Steps
        if load_from_file:
            data_loader = self.Get_Data_Loader()
            data_list   = data_loader.Read_Data( test_file_path, keep_in_memory = False, lowercase = lowercase )

            if len( data_list ) == 0:
                self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Error: Test Data List Is Empty" )
                return -1

            # Assumes NER-CL Model / DataLoader - TODO: Fix Function Parameters To Match Previous Calls.
            encoded_input, encoded_output = data_loader.Encode_Model_Data( data_list, use_csr_format = self.Get_Model().Get_Use_CSR_Format(),
                                                                           term_sequence_only = True, concept_delimiter = ",", mask_term_sequence = False,
                                                                           separate_sentences = True, restrict_context = False, label_per_sub_word = True )

        # Perform Concept Linking Inference Over Inputs
        model_predictions = self.Predict_Multi_Task_NER_CL( encoded_input = encoded_input, return_vector = True, return_raw_values = False )

        # Convert Matrix To Dense Format
        if isinstance( encoded_input,  COO ) or isinstance( encoded_input,  csr_matrix ): encoded_input  = encoded_input.todense()
        if isinstance( encoded_output, COO ) or isinstance( encoded_output, csr_matrix ): encoded_output = encoded_output.todense()
        if isinstance( encoded_input,  np.matrixlib.matrix ): encoded_input  = np.asarray( encoded_input  )
        if isinstance( encoded_output, np.matrixlib.matrix ): encoded_output = np.asarray( encoded_output )

        # Compute Actual Instance Sub-Word Lengths
        actual_instance_sizes = []

        if len( actual_instance_sizes ) == 0 and "bert" in self.model.Get_Network_Model():
            # Use Input Sub-Word IDs To Determine Actual Lengths
            for sub_word_instance_input_ids in encoded_input[0]:
                non_padding_sub_words = [ sub_word_id for sub_word_id in sub_word_instance_input_ids if sub_word_id != self.Get_Data_Loader().Get_Sub_Word_PAD_Token_ID() ]
                actual_instance_sizes.append( len( non_padding_sub_words ) )

        metrics = self.Generate_Classification_Report( model_predictions = model_predictions, true_labels = encoded_output, actual_instance_sizes = actual_instance_sizes )

        # Compute Elapsed Time
        elapsed_time = "{:.2f}".format( time.time() - start_time )
        self.Print_Log( "NERLink::Evaluate_NER_Concept_Linking() - Elapsed Time: " + str( elapsed_time ) + " secs" )

        return metrics


    ############################################################################################
    #                                                                                          #
    #    Model Support Functions                                                               #
    #                                                                                          #
    ############################################################################################

    """
        Generates A Classification Report Between Model Predictions And Ground Truth Labels.
          Forwards Parameters To NER Or Concept Linking Specific Functions.

        Inputs:
            model_predictions      : Batch Of Model Prediction Instances (List or Numpy Array)
            true_labels            : Batch Of Ground Truth Label Instances (List or Numpy Array)
            actual_instance_sizes  : List Corresponding To Size Of Each NER/CL Sequence Instance In The Batch

        Outputs:
            metrics                : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Generate_Classification_Report( self, model_predictions, true_labels, actual_instance_sizes = [] ):
        if self.is_ner_model:
            self.Print_Log( "NERLink::Generate_Classification_Report() - Generating Classification Report For NER Model" )
            return self.Generate_NER_Classification_Report( model_predictions = model_predictions, true_labels = true_labels, actual_instance_sizes = actual_instance_sizes )
        elif self.is_concept_linking_model:
            self.Print_Log( "NERLink::Generate_Classification_Report() - Generating Classification Report For Concept Linking Model" )
            if self.model.Get_Loss_Function() == "sparse_categorical_crossentropy":
                return self.Generate_Sparse_Concept_Linking_Classification_Report( model_predictions = model_predictions, true_labels = true_labels, actual_instance_sizes = actual_instance_sizes )
            else:
                return self.Generate_Concept_Linking_Classification_Report( model_predictions = model_predictions, true_labels = true_labels, actual_instance_sizes = actual_instance_sizes )
        elif self.is_ner_cl_multi_task_model:
            # Check(s)
            if model_predictions and len( model_predictions ) < 2:
                self.Print_Log( "NERLink::Generate_Classification_Report() - Error: 'model_predictions' Does Not Contains NER+CL Predictions" )
                return None

            if true_labels and len( true_labels ) < 2:
                self.Print_Log( "NERLink::Generate_Classification_Report() - Error: 'true_labels' Does Not Contains NER+CL Labels" )
                return None

            ner_predictions, cl_predictions = model_predictions
            ner_labels, cl_labels           = true_labels
            ner_metrics, cl_metrics         = None, None

            if isinstance( ner_predictions, COO ) or isinstance( ner_predictions, csr_matrix ): ner_predictions = ner_predictions.todense()
            if isinstance( cl_predictions,  COO ) or isinstance( cl_predictions,  csr_matrix ): cl_predictions  = cl_predictions.todense()
            if isinstance( ner_labels,      COO ) or isinstance( ner_labels,      csr_matrix ): ner_labels      = ner_labels.todense()
            if isinstance( cl_labels,       COO ) or isinstance( cl_labels,       csr_matrix ): cl_labels       = cl_labels.todense()

            self.Print_Log( "NERLink::Generate_Classification_Report() - Generating NER Classification Report" )
            ner_metrics = self.Generate_NER_Classification_Report( model_predictions = ner_predictions, true_labels = ner_labels, actual_instance_sizes = actual_instance_sizes )

            self.Print_Log( "NERLink::Generate_Classification_Report() - Generating Classification Report For Concept Linking Model" )
            if self.model.Get_Loss_Function() == "sparse_categorical_crossentropy":
                cl_metrics = self.Generate_Sparse_Concept_Linking_Classification_Report( model_predictions = cl_predictions, true_labels = cl_labels, actual_instance_sizes = actual_instance_sizes )
            else:
                cl_metrics = self.Generate_Concept_Linking_Classification_Report( model_predictions = cl_predictions, true_labels = cl_labels )

            return ( ner_metrics, cl_metrics )



    """
        Generates Classification Report For NER

        Inputs:
            model_predictions      : Batch Of Model Prediction Instances (List or Numpy Array)
            true_labels            : Batch Of Ground Truth Label Instances (List or Numpy Array)
            actual_instance_sizes  : List Corresponding To Size Of Each NER/CL Sequence Instance In The Batch

        Outputs:
            metrics                : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Generate_NER_Classification_Report( self, model_predictions, true_labels, actual_instance_sizes = [] ):
        # Prepare Variables For Confusion Matrix/Metric Computations
        annotation_token_to_idx = self.Get_Data_Loader().Get_Annotation_Labels()
        idx_to_annotation_token = { v:k for k, v in annotation_token_to_idx.items() }
        model_correct_predictions, model_total_predictions, number_of_annotation_labels = 0, 0, len( self.Get_Data_Loader().Get_Annotation_Labels() )
        num_of_correct_tp_predictions, total_number_of_tp_instances = 0, 0
        confusion_matrix = [ [ 0 for _ in range( number_of_annotation_labels )] for _ in range( number_of_annotation_labels ) ]
        support_values   = { index : 0 for index in range( number_of_annotation_labels ) }

        # Compute Over All Tokens If 'actual_instance_sizes' Not Specified
        if len( actual_instance_sizes ) == 0: actual_instance_sizes = [ self.Get_Data_Loader().Get_Max_Sequence_Length() for _ in range( len( model_predictions ) ) ]

        # Compute Confusion Matrix Using Model Predictions And Ground Truth Labels
        for model_prediction_instance, true_token_instance, number_of_tokens in zip( model_predictions, true_labels, actual_instance_sizes ):
            if isinstance( model_prediction_instance, np.ndarray ) and ( model_prediction_instance.dtype == np.int32 or model_prediction_instance.dtype == np.float32 ):
                model_prediction_instance = self.Decode_NER_Output_Instance( model_prediction_instance )

            if isinstance( true_token_instance, np.ndarray ) and ( true_token_instance.dtype == np.int32 or true_token_instance.dtype == np.float32 ):
                true_token_instance = self.Decode_NER_Output_Instance( true_token_instance )

            for index, ( predicted_token, true_token ) in enumerate( zip( model_prediction_instance, true_token_instance ) ):
                # Convert Predicted/True Text Labels To Indices If Specified
                class_prediction_value = int( annotation_token_to_idx[predicted_token] ) if isinstance( predicted_token, str ) else predicted_token
                class_true_value       = int( annotation_token_to_idx[true_token] )      if isinstance( true_token,      str ) else true_token

                # Store Counts In Confusion Matrix
                confusion_matrix[class_true_value][class_prediction_value] += 1

                # Check If Prediction Is Correct Prediction
                if class_prediction_value == class_true_value: model_correct_predictions += 1

                # Increment Number Of Total Processed Tokens
                model_total_predictions += 1

                # Increment Per Class Support Values
                support_values[class_true_value] += 1

                # We Do Not Want To Include Counts For Padding Tokens
                if index > number_of_tokens - 1: break

                # TP Accuracy Counts
                if class_true_value > 0:
                    # Count Number Of Correct TP Predictions
                    if class_prediction_value == class_true_value: num_of_correct_tp_predictions += 1

                    # Count Total Number Of Correct TP Predictions
                    total_number_of_tp_instances += 1

        # Metrics
        metrics = { k : {} for k, v in annotation_token_to_idx.items() }

        ################################
        # Add Per Class Support Values #
        ################################
        for class_idx in annotation_token_to_idx.values():
            metrics[idx_to_annotation_token[class_idx]]["Support"] = support_values[class_idx]

        #################################
        # Compute Precision (Per Class) #
        #################################
        for class_idx in annotation_token_to_idx.values():
            if class_idx < 0: continue

            tp    = confusion_matrix[class_idx][class_idx]
            tp_fp = 0

            for row_idx in annotation_token_to_idx.values():
                tp_fp += confusion_matrix[row_idx][class_idx]

            class_precision = float( tp / tp_fp ) if tp > 0 and tp_fp > 0 else 0.0
            metrics[idx_to_annotation_token[class_idx]]["Precision"] = class_precision

        ##############################
        # Compute Recall (Per Class) #
        ##############################
        for class_idx in annotation_token_to_idx.values():
            if class_idx < 0: continue

            tp    = confusion_matrix[class_idx][class_idx]
            tp_fn = 0

            for column_idx in annotation_token_to_idx.values():
                tp_fn += confusion_matrix[class_idx][column_idx]

            class_recall = float( tp / tp_fn ) if tp > 0 and tp_fn > 0 else 0.0
            metrics[idx_to_annotation_token[class_idx]]["Recall"]  = class_recall

        ############################################
        # Compute F1-Score / F-Measure (Per Class) #
        ############################################
        for label_reference in metrics:
            precision = metrics[label_reference]["Precision"]
            recall    = metrics[label_reference]["Recall"]

            if precision > 0 and recall > 0:
                metrics[label_reference]["F1_Score"] = ( 2 * precision * recall ) / ( precision + recall )
            else:
                metrics[label_reference]["F1_Score"] = 0.0

        #############################################
        # Compute Overal Averages Among All Metrics #
        #############################################
        avg_precision, avg_recall, avg_f1_score, total_support, num_labels_with_support = 0.0, 0.0, 0.0, 0, 0

        for label in metrics.keys():
            if metrics[label]["Support"] > 0:
                avg_precision += metrics[label]["Precision"]
                avg_recall    += metrics[label]["Recall"]
                avg_f1_score  += metrics[label]["F1_Score"]
                total_support += metrics[label]["Support"]
                num_labels_with_support += 1

        metrics["Avg / Total"] = {}
        metrics["Avg / Total"]["Precision"] = float( avg_precision / num_labels_with_support )
        metrics["Avg / Total"]["Recall"]    = float( avg_recall    / num_labels_with_support )
        metrics["Avg / Total"]["F1_Score"]  = float( avg_f1_score  / num_labels_with_support )
        metrics["Avg / Total"]["Support"]   = total_support

        ############################
        # Compute Overall Accuracy #
        ############################
        metrics["Accuracy (Model)"] = float( model_correct_predictions     / model_total_predictions )
        metrics["Accuracy (Data)"]  = float( num_of_correct_tp_predictions / total_number_of_tp_instances )

        ##########################
        # Print Confusion Matrix #
        ##########################
        self.Print_Log( "", force_print = True )

        title_str  = '{:^80s}'.format( "Confusion Matrix" )
        self.Print_Log( str( title_str ), force_print = True )

        # Print Confusion Matrix Column Legend
        self.Print_Log( str( '{:<20s}'.format( "" ) ), print_new_line = False, force_print = True )

        for column_idx in range( len( confusion_matrix ) ):
            prediction_column_name = '{:<20s}'.format( idx_to_annotation_token[column_idx] + " (Pred)" )
            self.Print_Log( str( prediction_column_name ), print_new_line = False, force_print = True )

        self.Print_Log( "", force_print = True  )

        # Print Confusion Matrix Row Legend And Values
        confusion_matrix_str = ""

        for row_idx in range( len( confusion_matrix ) ):
            confusion_matrix_str  += '{:20s}'.format( idx_to_annotation_token[row_idx] + " (True)" )

            for column_idx in range( len( confusion_matrix ) ):
                confusion_matrix_str += '{:<20d}'.format( confusion_matrix[row_idx][column_idx] )

            confusion_matrix_str += "\n"

        self.Print_Log( str( confusion_matrix_str ), force_print = True )

        ###############################
        # Print Classification Report #
        ###############################
        self.Print_Log( "", force_print = True )
        title = list( metrics.keys() )
        if "Accuracy" in title: title.remove( "Accuracy" )

        title_str  = '{:^100s}'.format( "Classification Report" )
        self.Print_Log( str( title_str ), force_print = True )

        header_str = '{:20s} {:20s} {:20s} {:20s} {:20s}'.format( "Class", "Precision", "Recall", "F1-Score", "Support" )
        self.Print_Log( str( header_str ), force_print = True )

        metric_str = ""
        for id in title:
            if not isinstance( metrics[id], dict ): continue
            metric_str += '{:<20s} {:<20.2f} {:<20.2f} {:<20.2f} {:<20d}'.format( id, metrics[id]["Precision"], metrics[id]["Recall"], metrics[id]["F1_Score"], metrics[id]["Support"] ) + "\n"

        self.Print_Log( str( metric_str ), force_print = True )

        self.Print_Log( "\nOverall Data Accuracy : " + str( metrics["Accuracy (Data)"] ) + "\n" )
        self.Print_Log( "\nOverall Model Accuracy: " + str( metrics["Accuracy (Model)"] ) + "\n" )

        return metrics

    """
        Generates Classification Report For Concept Linking

        Inputs:
            model_predictions      : Batch Of Model Prediction Instances (List or Numpy Array)
            true_labels            : Batch Of Ground Truth Label Instances (List or Numpy Array)
            actual_instance_sizes  : List Corresponding To Size Of Each NER/CL Sequence Instance In The Batch

        Outputs:
            metrics                : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Generate_Concept_Linking_Classification_Report( self, model_predictions, true_labels, actual_instance_sizes = [] ):
        # Prepare Variables For Confusion Matrix/Metric Computations
        concept_token_to_idx = self.Get_Data_Loader().Get_Concept_ID_Dictionary()

        # Remove Padding Token From Concept ID Dictionary
        # _ = concept_token_to_idx.pop( self.Get_Data_Loader().Get_Padding_Token() )
        idx_to_concept_token = { v:k for k, v in concept_token_to_idx.items() }
        model_correct_predictions, model_total_predictions, number_of_concept_labels = 0, 0, len( concept_token_to_idx )
        num_of_correct_tp_predictions, total_number_of_tp_instances = 0, 0
        confusion_matrices   = [ [ [ 0 for _ in range( 2 ) ] for _ in range( 2 ) ] for _ in range( number_of_concept_labels ) ]
        support_values       = { index : 0 for index in range( number_of_concept_labels ) }
        labels_to_skip       = [ class_id for class_id, value in concept_token_to_idx.items() if value < 0 ]

        # If 'actual_instances_sizes' Contains No Data, Generate Some False Data To Let The Process Proceed
        if len( actual_instance_sizes ) == 0:
            if model_predictions.ndim == 1:
                actual_instance_sizes = [ 1 for _ in range( model_predictions.shape[0] ) ]
            else: # i.e. model_predictions.ndim == 2:
                actual_instance_sizes = [ model_predictions.shape[1] for _ in range( model_predictions.shape[0] ) ]


        # Compute Confusion Matrix Using Model Predictions And Ground Truth Labels
        for prediction_instance, true_instance, number_of_tokens in zip( model_predictions, true_labels, actual_instance_sizes ):
            if not isinstance( prediction_instance, np.ndarray ): prediction_instance = np.asarray( prediction_instance )
            if not isinstance( true_instance,       np.ndarray ): true_instance       = np.asarray( true_instance )

            # Instances Are Multi-Class/Label Per Sub-Word Token
            if prediction_instance.ndim == 2 or true_instance.ndim == 2 :
                for token_instance_idx in range( prediction_instance.shape[0] ):
                    for class_idx in range( prediction_instance.shape[1] ):
                        class_prediction_value = int( prediction_instance[token_instance_idx][class_idx] )
                        class_true_value       = int( true_instance[token_instance_idx][class_idx] )

                        # Store Counts In Confusion Matrix
                        confusion_matrices[class_idx][class_true_value][class_prediction_value] += 1

                        # Check If Prediction Is Correct Prediction (Accuracy - Model - Includes TN/Zeros)
                        if class_prediction_value == class_true_value: model_correct_predictions += 1

                        # Increment Number Of Total Processed Tokens (Accuracy - Model - Includes TN/Zeros)
                        model_total_predictions += 1

                        # Increment Per Class Support And TP Accuracy Counts (Accuracy - Data - Exclude TN/Zeros)
                        if class_true_value == 1:
                            # Count Number Of Correct TP Predictions
                            if class_prediction_value == class_true_value: num_of_correct_tp_predictions += 1

                            # Count Total Number Of Correct TP Predictions
                            total_number_of_tp_instances += 1

                            # Increment Per Class Support Values
                            support_values[class_idx] += 1

                    if token_instance_idx >= number_of_tokens:
                        break

            # Instances Are Multi-Class/Label Or Multi-Class/Label Per Sub-Word Token
            #   Examine All Elements In Prediction & True Instances To Determine Model Prediction Accuracy
            #   Compute Confusion Matrix For Precision, Recall & F1-Score Metrics In Addition To Accuracy
            else:
                for class_idx in range( prediction_instance.shape[0] ):
                    class_prediction_value = int( prediction_instance[class_idx] )
                    class_true_value       = int( true_instance[class_idx] )

                    # Store Counts In Confusion Matrix
                    confusion_matrices[class_idx][class_true_value][class_prediction_value] += 1

                    # Check If Prediction Is Correct Prediction (Accuracy - Model - Includes TN/Zeros)
                    if class_prediction_value == class_true_value: model_correct_predictions += 1

                    # Increment Number Of Total Processed Tokens (Accuracy - Model - Includes TN/Zeros)
                    model_total_predictions += 1

                    # Increment Per Class Support And TP Accuracy Counts (Accuracy - Data - Exclude TN/Zeros)
                    if class_true_value == 1:
                        # Count Number Of Correct TP Predictions
                        if class_prediction_value == class_true_value: num_of_correct_tp_predictions += 1

                        # Count Total Number Of Correct TP Predictions
                        total_number_of_tp_instances += 1

                        # Increment Per Class Support Values
                        support_values[class_idx] += 1

        # Metrics
        metrics             = { k : {} for k, v in concept_token_to_idx.items() }
        fp_counts_per_class = [ confusion_matrix[0][1] for confusion_matrix in confusion_matrices ]
        fn_counts_per_class = [ confusion_matrix[1][0] for confusion_matrix in confusion_matrices ]
        tp_counts_per_class = [ confusion_matrix[1][1] for confusion_matrix in confusion_matrices ]

        global_tp, global_fp, global_fn = np.sum( tp_counts_per_class ), np.sum( fp_counts_per_class ), np.sum( fn_counts_per_class )

        #######################################################################################
        # Compute Per Class Precision, Recall & F1-Scores (Multi-Class, Multi-Label Specific) #
        #######################################################################################
        for class_token, class_idx in concept_token_to_idx.items():
            if class_token in labels_to_skip: continue

            tp    = confusion_matrices[class_idx][1][1]
            tp_fp = tp + confusion_matrices[class_idx][0][1]
            tp_fn = tp + confusion_matrices[class_idx][1][0]
            class_precision = float( tp / tp_fp ) if tp > 0 and tp_fp > 0 else 0.0
            class_recall    = float( tp / tp_fn ) if tp > 0 and tp_fn > 0 else 0.0
            class_f1_score  = 0.0

            if class_precision > 0 and class_recall > 0:
                class_f1_score = ( 2 * class_precision * class_recall ) / ( class_precision + class_recall )

            metrics[idx_to_concept_token[class_idx]]["Precision"] = class_precision
            metrics[idx_to_concept_token[class_idx]]["Recall"]    = class_recall
            metrics[idx_to_concept_token[class_idx]]["F1_Score"]  = class_f1_score

            # Add Per Class Support Values
            metrics[idx_to_concept_token[class_idx]]["Support"]   = support_values[class_idx]

        #############################################
        # Compute Overal Averages Among All Metrics #
        #############################################
        macro_avg_precision, macro_avg_recall, macro_avg_f1_score                      = 0.0, 0.0, 0.0
        macro_actual_avg_precision, macro_actual_avg_recall, macro_actual_avg_f1_score = 0.0, 0.0, 0.0
        weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score             = 0.0, 0.0, 0.0
        total_support, num_labels_with_support                                         = 0, 0

        # Aggregate Precision, Recall, F1-Score And Support Metrics/Counts For Each Class/Label
        for label in metrics.keys():
            if label in labels_to_skip: continue

            # Aggregate 'Macro' Counts
            macro_avg_precision += metrics[label]["Precision"]
            macro_avg_recall    += metrics[label]["Recall"]
            macro_avg_f1_score  += metrics[label]["F1_Score"]

            # Aggregate 'Macro_Actual' Counts
            if metrics[label]["Support"] > 0:
                macro_actual_avg_precision += metrics[label]["Precision"]
                macro_actual_avg_recall    += metrics[label]["Recall"]
                macro_actual_avg_f1_score  += metrics[label]["F1_Score"]

            # Aggregate 'Weighted' Counts
            weighted_avg_precision += metrics[label]["Precision"] * metrics[label]["Support"]
            weighted_avg_recall    += metrics[label]["Recall"]    * metrics[label]["Support"]
            weighted_avg_f1_score  += metrics[label]["F1_Score"]  * metrics[label]["Support"]

            # Aggregate Total Support For Each Class/Label
            total_support += metrics[label]["Support"]

            if metrics[label]["Support"] > 0: num_labels_with_support += 1

        metrics["Avg / Total"] = {}

        # Compute 'Micro' Metrics
        metrics["Avg / Total"]["Micro_Precision"] = float( global_tp / ( global_tp + global_fp ) )
        metrics["Avg / Total"]["Micro_Recall"]    = float( global_tp / ( global_tp + global_fn ) )

        if metrics["Avg / Total"]["Micro_Precision"] == 0 or metrics["Avg / Total"]["Micro_Recall"] == 0:
            metrics["Avg / Total"]["Micro_F1_Score"] = 0.0
        else:
            metrics["Avg / Total"]["Micro_F1_Score"] = float( ( 2 * metrics["Avg / Total"]["Micro_Precision"] * metrics["Avg / Total"]["Micro_Recall"] ) /
                                                              ( metrics["Avg / Total"]["Micro_Precision"] + metrics["Avg / Total"]["Micro_Recall"] ) )

        # Macro Divides The Precision, Recall And F1-Scores By The Number Of Classes/Labels
        metrics["Avg / Total"]["Macro_Precision"] = float( macro_avg_precision / len( concept_token_to_idx ) )
        metrics["Avg / Total"]["Macro_Recall"]    = float( macro_avg_recall    / len( concept_token_to_idx ) )
        metrics["Avg / Total"]["Macro_F1_Score"]  = float( macro_avg_f1_score  / len( concept_token_to_idx ) )

        # Compute The Actual Macro By Dividing Precision, Recall And F1-Scores By The Number Of Classes/Labels With Support > 0
        metrics["Avg / Total"]["Macro_Actual_Precision"] = float( macro_avg_precision / num_labels_with_support )
        metrics["Avg / Total"]["Macro_Actual_Recall"]    = float( macro_avg_recall    / num_labels_with_support )
        metrics["Avg / Total"]["Macro_Actual_F1_Score"]  = float( macro_avg_f1_score  / num_labels_with_support )

        # Weight Each Class Precision, Recall And F1-Scores By Their Support
        metrics["Avg / Total"]["Weighted_Precision"] = float( weighted_avg_precision / total_support )
        metrics["Avg / Total"]["Weighted_Recall"]    = float( weighted_avg_recall    / total_support )
        metrics["Avg / Total"]["Weighted_F1_Score"]  = float( weighted_avg_f1_score  / total_support )

        metrics["Avg / Total"]["Support"] = total_support

        ############################
        # Compute Overall Accuracy #
        ############################
        # Note: 'Accuracy (Model)' Measures The Model's Ability To Predict Correct TP and TN Instances. (Model Evaluation)
        #          i.e. y_pred = 0 & y_true = 0 / y_pred = 1 & y_true = 1 Are Both Correct Scenarios
        #       'Accuracy (Data)' Measures The Model's Ability To Predict Correct TP Instances. (Data Evaluation)
        #          i.e. y_pred = 1 & y_true = 1 Is The Only Correct Scenario.
        metrics["Accuracy (Model)"] = float( model_correct_predictions     / model_total_predictions )
        metrics["Accuracy (Data)"]  = float( num_of_correct_tp_predictions / total_number_of_tp_instances )

        # TODO - Fix Printing Confusion Matrix (These Will Be Really Large Though)

        ##########################
        # Print Confusion Matrix #
        ##########################
        # self.Print_Log( "", force_print = True )

        # title_str  = '{:^80s}'.format( "Confusion Matrix" )
        # self.Print_Log( str( title_str ), force_print = True )

        # # Print Confusion Matrix Column Legend
        # self.Print_Log( str( '{:<20s}'.format( "" ) ), print_new_line = False, force_print = True )

        # for column_idx in range( len( confusion_matrix ) ):
        #     prediction_column_name = '{:<20s}'.format( idx_to_concept_token[column_idx] + " (Pred)" )
        #     self.Print_Log( str( prediction_column_name ), print_new_line = False, force_print = True )

        # self.Print_Log( "", force_print = True  )

        # # Print Confusion Matrix Row Legend And Values
        # confusion_matrix_str = ""

        # for row_idx in range( len( confusion_matrix ) ):
        #     confusion_matrix_str  += '{:20s}'.format( idx_to_concept_token[row_idx] + " (True)" )

        #     for column_idx in range( len( confusion_matrix ) ):
        #         confusion_matrix_str += '{:<20d}'.format( confusion_matrix[row_idx][column_idx] )

        #     confusion_matrix_str += "\n"

        # self.Print_Log( str( confusion_matrix_str ), force_print = True )

        ###############################
        # Print Classification Report #
        ###############################
        # self.Print_Log( "", force_print = True )
        # title = list( metrics.keys() )
        # if "Accuracy" in title: title.remove( "Accuracy" )

        # title_str  = '{:^100s}'.format( "Classification Report" )
        # self.Print_Log( str( title_str ), force_print = True )

        # header_str = '{:20s} {:20s} {:20s} {:20s} {:20s}'.format( "Class", "Precision", "Recall", "F1-Score", "Support" )
        # self.Print_Log( str( header_str ), force_print = True )

        # metric_str = ""
        # for id in title:
        #     if not isinstance( metrics[id], dict ): continue
        #     metric_str += '{:<20s} {:<20.2f} {:<20.2f} {:<20.2f} {:<20d}'.format( id, metrics[id]["Precision"], metrics[id]["Recall"], metrics[id]["F1_Score"], metrics[id]["Support"] ) + "\n"

        # self.Print_Log( str( metric_str ), force_print = True )

        self.Print_Log( "\nOverall Data Accuracy : " + str( metrics["Accuracy (Data)"] ) + "\n" )
        self.Print_Log( "\nOverall Model Accuracy: " + str( metrics["Accuracy (Model)"] ) + "\n" )

        return metrics

    """
        Generates Classification Report For Concept Linking (For 'sparse_categorical_crossentropy' Loss Function)

        Inputs:
            model_predictions      : Batch Of Model Prediction Instances (List or Numpy Array)
            true_labels            : Batch Of Ground Truth Label Instances (List or Numpy Array)
            actual_instance_sizes  : List Corresponding To Size Of Each NER/CL Sequence Instance In The Batch

        Outputs:
            metrics                : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Generate_Sparse_Concept_Linking_Classification_Report( self, model_predictions, true_labels, actual_instance_sizes = [] ):
        # Prepare Variables For Confusion Matrix/Metric Computations
        concept_token_to_idx = self.Get_Data_Loader().Get_Concept_ID_Dictionary()

        # Remove Padding Token From Concept ID Dictionary
        # _ = concept_token_to_idx.pop( self.Get_Data_Loader().Get_Padding_Token() )
        idx_to_concept_token = { v:k for k, v in concept_token_to_idx.items() }
        model_correct_predictions, model_total_predictions, number_of_concept_labels = 0, 0, len( concept_token_to_idx )
        num_of_correct_tp_predictions, total_number_of_tp_instances = 0, 0
        confusion_matrix     = [ [ 0 for _ in range( number_of_concept_labels ) ] for _ in range( number_of_concept_labels ) ]
        support_values       = { index : 0 for index in range( number_of_concept_labels ) }

         # Compute Over All Tokens If 'actual_instance_sizes' Not Specified - (Not Working Currently)
        if len( actual_instance_sizes ) == 0:
            for true_instance in true_labels:
                temp_list = [ val for val in true_instance if val != 0 ]
                actual_instance_sizes.append( len( temp_list ) )

        # Compute Confusion Matrix Using Model Predictions And Ground Truth Labels
        for prediction_instance, true_instance, number_of_tokens in zip( model_predictions, true_labels, actual_instance_sizes ):
            if not isinstance( prediction_instance, np.ndarray ): prediction_instance = np.asarray( prediction_instance )
            if not isinstance( true_instance,       np.ndarray ): true_instance       = np.asarray( true_instance )

            # Examine All Elements In Prediction & True Instances To Determine Model Prediction Accuracy
            #   Compute Confusion Matrix For Precision, Recall & F1-Score Metrics In Addition To Accuracy
            for index, ( class_prediction_value, class_true_value ) in enumerate( zip( prediction_instance, true_instance ) ):
                # Store Counts In Confusion Matrix
                confusion_matrix[class_true_value][class_prediction_value] += 1

                # Check If Prediction Is Correct Prediction (Accuracy - Model - Includes TN/Zeros)
                if class_prediction_value == class_true_value: model_correct_predictions += 1

                # Increment Number Of Total Processed Tokens (Accuracy - Model - Includes TN/Zeros)
                model_total_predictions += 1

                # Increment Per Class Support Values
                support_values[class_true_value] += 1

                # Increment Per Class Support And TP Accuracy Counts (Accuracy - Data - Exclude TN/Zeros)
                if class_true_value > concept_token_to_idx[self.data_loader.Get_CUI_LESS_Token()] and \
                   class_true_value > concept_token_to_idx[self.data_loader.Get_Padding_Token()]:
                    # Count Number Of Correct TP Predictions
                    if class_prediction_value == class_true_value: num_of_correct_tp_predictions += 1

                    # Count Total Number Of Correct TP Predictions
                    total_number_of_tp_instances += 1

                if index >= number_of_tokens: break

        # Metrics
        metrics = { k : {} for k, v in concept_token_to_idx.items() }

        ################################
        # Add Per Class Support Values #
        ################################
        for class_idx in concept_token_to_idx.values():
            if class_idx < 0: continue
            metrics[idx_to_concept_token[class_idx]]["Support"] = support_values[class_idx]

        #################################
        # Compute Precision (Per Class) #
        #################################
        for class_idx in concept_token_to_idx.values():
            if class_idx < 0: continue

            tp    = confusion_matrix[class_idx][class_idx]
            tp_fp = 0

            for row_idx in concept_token_to_idx.values():
                tp_fp += confusion_matrix[row_idx][class_idx]

            class_precision = float( tp / tp_fp ) if tp > 0 and tp_fp > 0 else 0.0
            metrics[idx_to_concept_token[class_idx]]["Precision"] = class_precision

        ##############################
        # Compute Recall (Per Class) #
        ##############################
        for class_idx in concept_token_to_idx.values():
            if class_idx < 0: continue

            tp    = confusion_matrix[class_idx][class_idx]
            tp_fn = 0

            for column_idx in concept_token_to_idx.values():
                tp_fn += confusion_matrix[class_idx][column_idx]

            class_recall = float( tp / tp_fn ) if tp > 0 and tp_fn > 0 else 0.0
            metrics[idx_to_concept_token[class_idx]]["Recall"]  = class_recall

        ############################################
        # Compute F1-Score / F-Measure (Per Class) #
        ############################################
        for label_reference in metrics:
            precision = metrics[label_reference]["Precision"]
            recall    = metrics[label_reference]["Recall"]

            if precision > 0 and recall > 0:
                metrics[label_reference]["F1_Score"] = ( 2 * precision * recall ) / ( precision + recall )
            else:
                metrics[label_reference]["F1_Score"] = 0.0

        #############################################
        # Compute Overal Averages Among All Metrics #
        #############################################
        avg_precision, avg_recall, avg_f1_score, total_support, num_labels_with_support = 0.0, 0.0, 0.0, 0, 0

        for label in metrics.keys():
            if metrics[label]["Support"] > 0:
                avg_precision += metrics[label]["Precision"]
                avg_recall    += metrics[label]["Recall"]
                avg_f1_score  += metrics[label]["F1_Score"]
                total_support += metrics[label]["Support"]
                num_labels_with_support += 1

        metrics["Avg / Total"] = {}
        metrics["Avg / Total"]["Precision"] = float( avg_precision / num_labels_with_support )
        metrics["Avg / Total"]["Recall"]    = float( avg_recall    / num_labels_with_support )
        metrics["Avg / Total"]["F1_Score"]  = float( avg_f1_score  / num_labels_with_support )
        metrics["Avg / Total"]["Support"]   = total_support

        ############################
        # Compute Overall Accuracy #
        ############################
        metrics["Accuracy (Model)"] = float( model_correct_predictions     / model_total_predictions )
        metrics["Accuracy (Data)"]  = float( num_of_correct_tp_predictions / total_number_of_tp_instances )

        ##########################
        # Print Confusion Matrix #
        ##########################
        self.Print_Log( "", force_print = True )

        title_str  = '{:^80s}'.format( "Confusion Matrix" )
        self.Print_Log( str( title_str ), force_print = True )

        # Print Confusion Matrix Column Legend
        self.Print_Log( str( '{:<20s}'.format( "" ) ), print_new_line = False, force_print = True )

        for column_idx in range( len( confusion_matrix ) ):
            prediction_column_name = '{:<20s}'.format( idx_to_concept_token[column_idx] + " (Pred)" )
            self.Print_Log( str( prediction_column_name ), print_new_line = False, force_print = True )

        self.Print_Log( "", force_print = True  )

        # Print Confusion Matrix Row Legend And Values
        confusion_matrix_str = ""

        for row_idx in range( len( confusion_matrix ) ):
            confusion_matrix_str  += '{:20s}'.format( idx_to_concept_token[row_idx] + " (True)" )

            for column_idx in range( len( confusion_matrix ) ):
                confusion_matrix_str += '{:<20d}'.format( confusion_matrix[row_idx][column_idx] )

            confusion_matrix_str += "\n"

        self.Print_Log( str( confusion_matrix_str ), force_print = True )

        ###############################
        # Print Classification Report #
        ###############################
        self.Print_Log( "", force_print = True )
        title = list( metrics.keys() )
        if "Accuracy" in title: title.remove( "Accuracy" )

        title_str  = '{:^100s}'.format( "Classification Report" )
        self.Print_Log( str( title_str ), force_print = True )

        header_str = '{:20s} {:20s} {:20s} {:20s} {:20s}'.format( "Class", "Precision", "Recall", "F1-Score", "Support" )
        self.Print_Log( str( header_str ), force_print = True )

        metric_str = ""
        for id in title:
            if not isinstance( metrics[id], dict ): continue
            metric_str += '{:<20s} {:<20.2f} {:<20.2f} {:<20.2f} {:<20d}'.format( id, metrics[id]["Precision"], metrics[id]["Recall"], metrics[id]["F1_Score"], metrics[id]["Support"] ) + "\n"

        self.Print_Log( str( metric_str ), force_print = True )

        self.Print_Log( "\nOverall Data Accuracy : " + str( metrics["Accuracy (Data)"] ) + "\n" )
        self.Print_Log( "\nOverall Model Accuracy: " + str( metrics["Accuracy (Model)"] ) + "\n" )

        return metrics

    """
        Computes Accuracy Between Model Predictions And Ground Truth Labels For Concept Linking.

        Notes: Not Tested With NER Model.
               This Assumes One-To-One Relationship Between Instances.
    """
    def Compute_Accuracy( self, model_prediction_instances, ground_truth_instances ):
        # Check(s)
        if isinstance( ground_truth_instances, COO        ): ground_truth_instances = ground_truth_instances.todense()
        if isinstance( ground_truth_instances, csr_matrix ): ground_truth_instances = ground_truth_instances.todense()

        correct_predictions, total_instances = 0, 0

        for prediction_instance, true_instance in zip( model_prediction_instances, ground_truth_instances ):
            if not isinstance( prediction_instance, np.ndarray ): prediction_instance = np.asarray( prediction_instance )
            if not isinstance( true_instance,       np.ndarray ): true_instance       = np.asarray( true_instance )

            if prediction_instance.max() > 0.0:
                prediction_idx = np.argmax( prediction_instance )
                true_idx       = np.argmax( true_instance )

                if prediction_idx == true_idx:
                    correct_predictions += 1
            else:
                self.Print_Log( "NERLink::Compute_Accuracy() - Warning: Prediction Instance Does Not Contain Element == 1 or 1.0" )

            total_instances += 1

        return float( correct_predictions / total_instances )

    """
        Computes Class Weights For Models.
           Weights Are Generated With Shape: ( number_of_labels )

        This Attributes More Or Less Importance To Training Labels.
           i.e. 'O' Labels Can Have Less Importance Than Other Labels. (Class Imbalance Mitigation)

        NOTE: Cannot Be Used With Sequence Labeling Models, Sample Weights Must Be Used For This.

        Inputs:
            number_of_labels  : Number Of Class Labels  (Integer)
            class_weight_dict : Label Weights Per Class (Dictionary)
                                 i.e. { 0 : 0.125, 1 : 1, 2 : 2.0, ..., 9 : 1 }

        Outputs:
            class_weight      : Class Weights Of Shape ( number_of_instances )
    """
    def Generate_Class_Weights( self, number_of_labels, class_weight_dict ):


        self.Print_Log( "NERLink::Generate_Class_Weights() - Generating Sample Weights" )
        self.Print_Log( "NERLink::Generate_Class_Weights() - Number Of Labels: " + str( number_of_labels ) )

        # Generate Class Weights
        class_weights = np.ones( shape = ( number_of_labels ) )

        for label_idx in range( class_weights.shape[0] ):
            if label_idx in class_weight_dict.keys():
                class_weights[label_idx] = class_weight_dict[label_idx]

        self.Print_Log( "NERLink::Generate_Class_Weights() - Sample Weights: " + str( class_weights ) )
        self.Print_Log( "NERLink::Generate_Class_Weights() - Complete" )

        return class_weights

    """
        Computes Sample Weight For Sequence Labeling Models.
           Weights Are Generated With Shape: ( number_of_instances, sequence_length )

        This Attributes More Or Less Importance To Training Instances.
           i.e. 'O' Label Instances Can Have Less Importance Than Other Labels. (Class Imbalance Mitigation)

        Inputs:
            encoded_outputs    : Encoded Training Instance Labels  (List, Numpy Array, csr_matrix or COO)
            sample_weight_dict : Label Weights Per Sample Instance (Dictionary)
                                 i.e. { 0 : 0.125, 1 : 1, 2 : 2.0, ..., 9 : 1 }

        Outputs:
            sample_weights     : Sample Weight Of Shape ( number_of_instances, sequence_length ) (Numpy array or COO matrix)
    """
    def Generate_Sample_Weights( self, encoded_outputs, sample_weight_dict, fill_value = 1, return_coo_matrix = False ):
        number_of_instances, sequence_length, convert_to_dense = 0, 0, False

        if isinstance( encoded_outputs, list ): encoded_outputs = np.asarray( encoded_outputs )

        if isinstance( encoded_outputs, COO ) or isinstance( encoded_outputs, csr_matrix ) or isinstance( encoded_outputs, np.ndarray ):
            if len( encoded_outputs.shape ) < 2:
                self.Print_Log( "NERLink::Generate_Sample_Weights() - Error: 'encoded_outputs' Dimensions < 2 / Unable To Generate Sequence Sample Weights", force_print = True )
                return None

            number_of_instances = encoded_outputs.shape[0]
            sequence_length     = encoded_outputs.shape[1]

            if isinstance( encoded_outputs, COO ) or isinstance( encoded_outputs, csr_matrix ): convert_to_dense = True

        self.Print_Log( "NERLink::Generate_Sample_Weights() - Generating Sample Weights" )
        self.Print_Log( "NERLink::Generate_Sample_Weights() - Number Of Instances: " + str( number_of_instances ) )
        self.Print_Log( "NERLink::Generate_Sample_Weights() - Sequence Length    : " + str( sequence_length ) )

        # Generate Sample Weights
        sample_weights, sample_weight_row, sample_weight_col, sample_weight_data = None, [], [], []
        if not return_coo_matrix: sample_weights = np.full( shape = ( number_of_instances, sequence_length ),
                                                            fill_value = fill_value, dtype = np.float32 )

        # Encoded Outputs In 'sparse_categorical_crossentropy' Loss Function Format
        if np.ndim( encoded_outputs ) == 2:
            for instance_idx, instance in enumerate( encoded_outputs ):
                if convert_to_dense: instance = instance.todense()

                for sequence_idx, label_idx in enumerate( instance ):
                    if label_idx in sample_weight_dict.keys():
                        if return_coo_matrix:
                            sample_weight_row.append( instance_idx )
                            sample_weight_col.append( sequence_idx )
                            sample_weight_data.append( sample_weight_dict[label_idx] )
                        else:
                            sample_weights[instance_idx][sequence_idx] = sample_weight_dict[label_idx]

        # Encoded Outputs In 'binary_crossentropy' or 'categorical_crossentropy' Loss Function Format
        elif np.ndim( encoded_outputs ) == 3:
            for instance_idx, instance in enumerate( encoded_outputs ):
                for sequence_idx, sequence_labels in enumerate( instance ):
                    if convert_to_dense: sequence_labels = sequence_labels.todense()

                    label_idx = np.argmax( sequence_labels ).item()

                    if label_idx in sample_weight_dict.keys():
                        if return_coo_matrix:
                            sample_weight_row.append( instance_idx )
                            sample_weight_col.append( sequence_idx )
                            sample_weight_data.append( sample_weight_dict[label_idx] )
                        else:
                            sample_weights[instance_idx][sequence_idx] = sample_weight_dict[label_idx]

        if return_coo_matrix:
            sample_weight_data = np.asarray( sample_weight_data, dtype = np.float32 )
            sample_weights = COO( [ sample_weight_row, sample_weight_col ], sample_weight_data, shape = ( number_of_instances, sequence_length ), fill_value = fill_value )

        self.Print_Log( "NERLink::Generate_Sample_Weights() - Sample Weights: " + str( sample_weights ) )
        self.Print_Log( "NERLink::Generate_Sample_Weights() - Complete" )

        return sample_weights

    """
        Reads Data From The File
    """
    def Read_Data( self, file_path, lowercase = True, keep_in_memory = True ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Read_Data( file_path, lowercase = lowercase, keep_in_memory = keep_in_memory )
        return None

    """
        Generates Unique Token/Concept Dictionaries From Specified Data List Or Data List Stored Inside DataLoader Instance
    """
    def Generate_Token_IDs( self, data_list = [], lowercase = True, update_dict = False, scale_embedding_weight_value = 1.0 ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Generate_Token_IDs( data_list = data_list, lowercase = lowercase, update_dict = update_dict,
                                                              scale_embedding_weight_value = scale_embedding_weight_value )
        return None

    """
        Vectorized/Binarized NER-CL Multi-Task Model Data - Used For Training/Evaluation Data
    """
    def Encode_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = True, keep_in_memory = True,
                           is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                           mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = True,
                           use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        if self.Get_Data_Loader() and isinstance( self.Get_Data_Loader(), BERTBioCreativeMultiTaskDataLoader ):
            return self.Get_Data_Loader().Encode_Model_Data( data_list = data_list, use_csr_format = use_csr_format, pad_input = pad_input, pad_output = pad_output,
                                                             keep_in_memory = keep_in_memory, is_validation_data = is_validation_data, is_evaluation_data = is_evaluation_data,
                                                             term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                             separate_sentences = separate_sentences, restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                             use_cui_less_labels = use_cui_less_labels, split_by_max_seq_length = split_by_max_seq_length, ignore_output_errors = ignore_output_errors )
        return None

    """
        Vectorized/Binarized NER Model Data - Used For Training/Evaluation Data
    """
    def Encode_NER_Model_Data( self, data_list = [], use_csr_format = False, keep_in_memory = True, number_of_threads = 4, is_validation_data = False, is_evaluation_data = False ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Encode_NER_Model_Data( data_list = data_list, use_csr_format = use_csr_format,
                                                                 keep_in_memory = keep_in_memory, number_of_threads = number_of_threads,
                                                                 is_validation_data = is_validation_data, is_evaluation_data = is_evaluation_data )
        return None

    """
        Vectorized/Binarized Concept Model Data - Used For Training/Evaluation Data
    """
    def Encode_CL_Model_Data( self, data_list = [], use_csr_format = False, pad_input = True, pad_output = True, keep_in_memory = True,
                              is_validation_data = False, is_evaluation_data = False, term_sequence_only = False, concept_delimiter = ",",
                              mask_term_sequence = False, separate_sentences = True, restrict_context = False, label_per_sub_word = True,
                              use_cui_less_labels = True, split_by_max_seq_length = True, ignore_output_errors = False ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Encode_CL_Model_Data( data_list = data_list, use_csr_format = use_csr_format, pad_input = pad_input, pad_output = pad_output,
                                                                keep_in_memory = keep_in_memory, is_validation_data = is_validation_data, is_evaluation_data = is_evaluation_data,
                                                                term_sequence_only = term_sequence_only, concept_delimiter = concept_delimiter, mask_term_sequence = mask_term_sequence,
                                                                separate_sentences = separate_sentences, restrict_context = restrict_context, label_per_sub_word = label_per_sub_word,
                                                                use_cui_less_labels = use_cui_less_labels, split_by_max_seq_length = split_by_max_seq_length, ignore_output_errors = ignore_output_errors )
        return None

    """
        Tokenize Model Data - Used For ELMo Implementation
    """
    def Tokenize_Model_Data( self, data_list = [], use_padding = True ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Tokenize_Model_Data( data_list = data_list, use_padding = use_padding )
        return None

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

            Inputs:

            Outputs:
    """
    def Encode_NER_Instance( self, text_sequence = "", annotations = [], annotation_labels = [], annotation_indices = [], concepts = {},
                             composite_mention_list = [], individual_mention_list = [] ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Encode_NER_Instance( text_sequence = text_sequence, annotations = annotations, annotation_labels = annotation_labels,
                                                               annotation_indices = annotation_indices, concepts = concepts, composite_mention_list = composite_mention_list,
                                                               individual_mention_list = individual_mention_list )
        return None

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

            Inputs:

            Outputs:
    """
    def Encode_CL_Instance( self, text_sequence = "", entry_term = "", annotation_concept = "", annotation_indices = None, pad_input = True, pad_output = False,
                            concept_delimiter = None, mask_term_sequence = False, separate_sentences = True, term_sequence_only = False,
                            restrict_context = False, label_per_sub_word = False ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Encode_CL_Instance( text_sequence = text_sequence, entry_term = entry_term, annotation_concept = annotation_concept,
                                                              annotation_indices = annotation_indices, pad_input = pad_input, pad_output = pad_output, concept_delimiter = concept_delimiter,
                                                              mask_term_sequence = mask_term_sequence, separate_sentences = separate_sentences, term_sequence_only = term_sequence_only,
                                                              restrict_context = restrict_context, label_per_sub_word = label_per_sub_word )
        return None

    """
        Decodes Sequence Of Token IDs to Sequence Of Token Strings

            Inputs:

            Outputs:
    """
    def Decode_NER_Input_Sequence( self, encoded_input_sequence = [], remove_padding = False ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_NER_Input_Instance( encoded_input_sequence, remove_padding = remove_padding )
        return None

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings
    """
    def Decode_NER_Output_Instance( self, encoded_output_sequence ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_NER_Output_Instance( encoded_output_sequence = encoded_output_sequence )
        return None

    """
        Decodes Output Sequence Of NER Label IDs To Sequence Of NER Label Strings
    """
    def Decode_NER_Instance( self, encoded_input_sequence, encoded_output_sequence, remove_padding = True ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_NER_Instance( encoded_input_sequence = encoded_input_sequence, encoded_output_sequence = encoded_output_sequence,
                                                               remove_padding = remove_padding )
        return None

    """
        Decodes Input & Output Sequence Of Concept Linking Token IDs And Concept ID Labels To Sequence Of Tokens & Concept ID Strings
    """
    def Decode_CL_Instance( self, encoded_input_instance, entry_term_mask = None, encoded_output_labels = [] ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_CL_Instance( encoded_input_instance = encoded_input_instance, entry_term_mask = entry_term_mask,
                                                              encoded_output_labels = encoded_output_labels )
        return None

    """
        Decodes Input Sequence Instance Of IDs To Entry Term String(s)
    """
    def Decode_CL_Input_Instance( self, encoded_input_instance, entry_term_mask = None ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_CL_Input_Instance( encoded_input_instance = encoded_input_instance, entry_term_mask = entry_term_mask )
        return None

    """
        Decodes Output Instance Of Labels For Concept Linking To List Of Concept ID Strings
    """
    def Decode_CL_Output_Instance( self, encoded_output_labels ):
        if self.Get_Data_Loader():
            return self.Get_Data_Loader().Decode_CL_Output_Instance( encoded_output_labels =  encoded_output_labels )
        return None

    """
        Loads The Model From A File

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Load_Model( self, model_path, model_name = "model", load_new_model = True ):
        self.Print_Log( "NERLink::Load_Model() - Loading Model From Path - " + str( model_path ), force_print = True )
        self.Print_Log( "NERLink::Load_Model() -         Model Name      - " + str( model_name ), force_print = True )

        if not re.search( r"\/$", model_path ): model_path += "/"

        # Check To See The Model Path Exists
        if not self.utils.Check_If_Path_Exists( model_path ):
            self.Print_Log( "NERLink::Load_Model() - Error: Specified Model Path Does Not Exist", force_print = True )
            return False

        self.Print_Log( "NERLink::Load_Model() - Fetching Network Model Type From Settings File" )
        network_model = self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "NetworkModelModel" )

        # Get Current Specified Device (GPU/CPU)
        use_gpu     = self.model.Get_Use_GPU()
        device_name = self.model.Get_Device_Name()

        self.Print_Log( "NERLink::Load_Model() - Detected Model Type: " + str( network_model ), force_print = True )

        # Load Network Architecture Type
        if network_model is not None:
            self.Print_Log( "NERLink::Load_Model() - Creating Model Data Loader" )

            # Reset Model Task Identifier Variable
            self.is_ner_model               = False
            self.is_concept_linking_model   = False
            self.is_ner_cl_multi_task_model = False

            # Create New DataLoader Instance With Options
            # Model Specific DataLoader Parameters/Settings
            if network_model in ["ner_bilstm", "ner_elmo", "concept_linking"]:
                self.data_loader = BioCreativeDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            elif network_model in ["ner_bert", "concept_linking_bert"]:
                self.data_loader = BERTBioCreativeDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            elif network_model in ["concept_linking_bert_distributed"]:
                self.data_loader = BERTDistributedBioCreativeDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            elif network_model in ["concept_linking_embedding_similarity"]:
                self.data_loader = MLPSimilarityDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            elif network_model in ["concept_linking_bert_embedding_similarity"]:
                self.data_loader = BERTSimilarityDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            elif network_model in ["ner_concept_linking_multi_task_bert"]:
                self.data_loader = BERTBioCreativeMultiTaskDataLoader( print_debug_log = self.debug_log, write_log_to_file = self.write_log, debug_log_file_handle = self.debug_log_file_handle )
            else:
                print( "NERLink::Load_Model() - Error Model \"" + str( network_model ) + "\"'s DataLoader Not Implemented" )
                raise NotImplementedError

            self.Print_Log( "NERLink::Load_Model() - Creating New \"" + str( network_model ) + "\" Model", force_print = True )

            custom_model_metrics = []
            self.model.Set_Debug_Log_File_Handle( None )

            if network_model == "ner_bilstm":
                self.is_ner_model = True
                custom_model_metrics = [ "F1_Score", "Precision", "Recall" ]
                self.model = BiLSTMModel( debug_log_file_handle = self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "ner_elmo":
                self.is_ner_model = True
                self.model = ELMoModel( debug_log_file_handle = self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "ner_bert":
                self.is_ner_model = True
                self.model = BERTModel( debug_log_file_handle = self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "concept_linking":
                self.is_concept_linking_model = True
                custom_model_metrics = [ "F1_Score", "Precision", "Recall" ]
                self.model = CLMLPModel( debug_log_file_handle = self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "concept_linking_bert":
                self.is_concept_linking_model = True
                custom_model_metrics = [ "F1_Score", "Precision", "Recall" ]
                self.model = CLBERTModel( debug_log_file_handle = self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "concept_linking_bert_distributed":
                self.is_concept_linking_model = True
                self.model = CLBERTModelDistributed( debug_log_file_handle =  self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "concept_linking_embedding_similarity":
                self.is_concept_linking_model = True
                self.model = CLMLPSimilarityModel( debug_log_file_handle =  self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "concept_linking_bert_embedding_similarity":
                self.is_concept_linking_model = True
                self.model = CLBERTSimilarityModel( debug_log_file_handle =  self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )
            elif network_model == "ner_concept_linking_multi_task_bert":
                self.is_ner_cl_multi_task_model = True
                self.model = NERCLMultiTaskModel( debug_log_file_handle =  self.debug_log_file_handle, use_gpu = use_gpu, device_name = device_name )

            # Load The Model From File & Model Settings To Model Object
            self.Print_Log( "NERLink::Load_Model() - Loading Model", force_print = True )

            self.model.Load_Model( model_path = model_path + model_name, load_new_model = load_new_model, model_metrics = custom_model_metrics, reinitialize_gpu = False )

            # Load Model Primary And Secondary Keys
            self.Print_Log( "NERLink::Load_Model() - Loading Model Token ID Dictionary", force_print = True )

            if self.utils.Check_If_File_Exists( model_path + model_name + "_token_id_key_data" ):
                self.Get_Data_Loader().Load_Token_ID_Key_Data( model_path + model_name + "_token_id_key_data" )
            else:
                self.Print_Log( "NERLink::Error: Model Token ID Key File Does Not Exist", force_print = True )

            # Check If Previous Model Utilized Embeddings To Train
            self.Print_Log( "NERLink::Load_Model() - Fetching DataLoader Embeddings From Settings File" )

            embedding_a_path, embedding_b_path = None, None

            if self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingAPath" ) != "":
                embedding_a_path = self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingAPath" )
            if self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingBPath" ) != "":
                embedding_b_path = self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingBPath" )

            # Load Embedding A File Or Set Simulate Embeddings Loaded Data Loader Flag
            if embedding_a_path is not None and self.utils.Check_If_File_Exists( file_path = embedding_a_path ):
                self.Print_Log( "NERLink::Load_Model() - Loading Embedding A Path From File: " + str( embedding_a_path ) )
                self.data_loader.Load_Embeddings( file_path = embedding_a_path, store_embeddings = True, location = "a" )
            else:
                self.Print_Log( "NERLink::Load_Model() - Embedding A File Not Found / Setting Simulate Embedding B = True: " + str( embedding_a_path ) )
                if self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingsALoaded" ) == "True":
                    self.Get_Data_Loader().Set_Simulate_Embeddings_A_Loaded_Mode( True )

            # Load Embedding B File Or Set Simulate Embeddings Loaded Data Loader Flag
            if embedding_b_path is not None and self.utils.Check_If_File_Exists( file_path = embedding_b_path ):
                self.Print_Log( "NERLink::Load_Model() - Loading Embedding A Path From File: " + str( embedding_b_path ) )
                self.data_loader.Load_Embeddings( file_path = embedding_b_path, store_embeddings = True, location = "b" )
            else:
                self.Print_Log( "NERLink::Load_Model() - Embedding A File Not Found / Setting Simulate Embedding B = True: " + str( embedding_b_path ) )
                if self.Get_Setting_Value_From_Model_Settings( model_path + model_name + "_settings.cfg", "EmbeddingsBLoaded" ) == "True":
                    self.Get_Data_Loader().Set_Simulate_Embeddings_B_Loaded_Mode( True )

            # Update Embeddings Based On Token ID And Concept ID Dictionaries
            if self.data_loader.Get_Number_Of_Unique_Tokens() > 0 or self.data_loader.Get_Number_Of_Concept_Elements() > 0 and \
                self.data_loader.Get_Number_Of_Embeddings_A() > 0 or self.data_loader.Get_Number_Of_Embeddings_B() > 0:
                self.data_loader.Update_Token_IDs()
            elif self.data_loader.Get_Number_Of_Unique_Tokens() == 0 and self.data_loader.Get_Number_Of_Concept_Elements() == 0 and \
                self.data_loader.Get_Number_Of_Embeddings_A() > 0 or self.data_loader.Get_Number_Of_Embeddings_B() > 0:
                self.Generate_Token_IDs()

            self.Print_Log( "NERLink::Load_Model() - Complete", force_print = True  )
            return True

        self.Print_Log( "NERLink::Load_Model() - Error Loading Model \"" + str( model_path + model_name ) + "\"" )
        return False

    """
        Saves Model To File

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model( self, model_path = "./", model_name = "model", save_format = "h5" ):
        # Check(s)
        if self.model.Get_Network_Model() == "ner_elmo":
            self.Print_Log( "NERLink::Save_Model() - Error: Unable To Save ELMo Model / Not Supported", force_print = True )
            return False

        if not self.Is_Model_Loaded():
            self.Print_Log( "NERLink::Save_Model() - Error: No Model Object In Memory / Has Model Been Trained or Loaded?", force_print = True )
            return False

        if not self.utils.Check_If_Path_Exists( model_path ):
            self.Print_Log( "NERLink::Save_Model() - Creating Model Save Path: " + str( model_path ) )
            self.utils.Create_Path( model_path )

        if not re.search( r"\/$", model_path ): model_path += "/"

        self.Print_Log( "NERLink::Save_Model() - Saving Model To Path: " + str( model_path ), force_print = True )
        self.model.Save_Model( model_path = model_path + model_name, save_format = save_format )

        # Save Model Keys
        self.Save_Model_Keys( key_path = model_path, model_name = model_name )

        self.Print_Log( "NERLink::Save_Model() - Complete" )

        return True

    """
        Save Model Token ID Key Data

        Inputs:
            file_path : File Path (String)

        Outputs:
            None
    """
    def Save_Model_Keys( self, key_path = "./", model_name = "model", file_name = "_token_id_key_data" ):
        path_contains_directories = self.utils.Check_If_Path_Contains_Directories( key_path )
        self.Print_Log( "NERLink::Save_Model_Keys() - Checking If Path Contains Directories: " + str( path_contains_directories ) )

        if self.utils.Check_If_Path_Exists( key_path ) == False:
            self.Print_Log( "NERLink::Save_Model_Keys() - Creating Model Key Save Path: " + str( key_path ) )
            self.utils.Create_Path( key_path )

        if not re.search( r"\/$", key_path ): key_path += "/"

        self.Print_Log( "NERLink::Save_Model_Keys() - Saving Model Keys To Path: " + str( key_path ) )
        self.data_loader.Save_Token_ID_Key_Data( key_path + model_name + file_name )

        self.Print_Log( "NERLink::Save_Model_Keys() - Complete" )

    def Generate_Model_Depiction( self, path = "./" ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "NERLink::Generate_Model_Depiction() - Error: No Model Object In Memory / Has Model Been Trained Or Loaded Yet?", force_print = True )
            return

        self.Print_Log( "NERLink::Generate_Model_Depiction() - Generating Model Depiction" )

        self.model.Generate_Model_Depiction( path )

        self.Print_Log( "NERLink::Generate_Model_Depiction() - Complete" )

    """
        Generates Plots (PNG Images) For Reported Metric Values During Each Training Epoch
    """
    def Generate_Model_Metric_Plots( self, path ):
        # Check
        if self.Is_Model_Loaded() == False:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Error: No Model Object In Memory / Has Model Been Trained Or Loaded Yet?", force_print = True )
            return

        self.utils.Create_Path( path )
        if not re.search( r"\/$", path ): path += "/"

        history = self.model.model_history.history

        if "loss" in history:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Plotting Training Set - Loss vs Epoch" )
            plt.plot( range( len( self.model.model_history.epoch ) ), history['loss'] )
            plt.title( "Training: Loss vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Loss" )
            plt.savefig( str( path ) + "training_epoch_vs_loss.png" )
            plt.clf()

        if "acc" in history or "accuracy" in history:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Plotting Training Set - Accuracy vs Epoch" )
            plt.plot( range( len( self.model.model_history.epoch ) ), history['accuracy'] if 'accuracy' in history else history['acc'] )
            plt.title( "Training: Accuracy vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Accuracy" )
            plt.savefig( str( path ) + "training_epoch_vs_accuracy.png" )
            plt.clf()

        if "Precision" in history:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Plotting Training Set - Precision vs Epoch" )
            plt.plot( range( len( self.model.model_history.epoch ) ), history['Precision'] )
            plt.title( "Training: Precision vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Precision" )
            plt.savefig( str( path ) + "training_epoch_vs_precision.png" )
            plt.clf()

        if "Recall" in history:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Plotting Training Set - Recall vs Epoch" )
            plt.plot( range( len( self.model.model_history.epoch ) ), history['Recall'] )
            plt.title( "Training: Recall vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "Recall" )
            plt.savefig( str( path ) + "training_epoch_vs_recall.png" )
            plt.clf()

        if "F1_Score" in history:
            self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Plotting Training Set - F1-Score vs Epoch" )
            plt.plot( range( len( self.model.model_history.epoch ) ), history['F1_Score'] )
            plt.title( "Training: F1-Score vs Epoch" )
            plt.xlabel( "Epoch" )
            plt.ylabel( "F1-Score" )
            plt.savefig( str( path ) + "training_epoch_vs_f1.png" )
            plt.clf()


        self.Print_Log( "NERLink::Generate_Model_Metric_Plots() - Complete" )


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Checks If Checkpoint Directory Exists And Creates It If Not Existing
    """
    def Create_Checkpoint_Directory( self ):
        self.Print_Log( "NERLink::Create_Checkpoint_Directory() - Checking If Model Save Directory Exists: \"" + str( self.checkpoint_directory ) + "\"", force_print = True )

        if self.utils.Check_If_Path_Exists( self.checkpoint_directory ) == False:
            self.Print_Log( "NERLink::Create_Checkpoint_Directory() - Creating Directory", force_print = True )
            os.mkdir( self.checkpoint_directory )
        else:
            self.Print_Log( "NERLink::Init() - Directory Already Exists", force_print = True )

    """
        Fetches Neural Model Type From File
    """
    def Get_Setting_Value_From_Model_Settings( self, file_path, setting_name ):
        model_settings_list = self.utils.Read_Data( file_path )

        for model_setting in model_settings_list:
            if re.match( r'^#', model_setting ) or model_setting == "": continue
            key, value = model_setting.split( "<:>" )
            if key == setting_name:
                return str( value )

        return None

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

    def Get_Model( self ):                          return self.model

    def Get_Network_Model( self ):                  return self.model.Get_Network_Model()

    def Get_Data( self ):                           return self.data_loader.Get_Data()

    def Get_Inputs( self ):                         return self.data_loader.Get_NER_Inputs()

    def Get_Outputs( self ):                        return self.data_loader.Get_NER_Outputs()

    def Get_Number_Of_Unique_Features( self ):      return self.data_loader.Get_Number_Of_Unique_Tokens()

    def Get_Number_Of_Input_Elements( self ):       return self.data_loader.Get_Number_Of_Input_Elements()

    def Get_Data_Loader( self ):                    return self.data_loader

    def Is_Embeddings_Loaded( self ):               return self.data_loader.Is_Embeddings_A_Loaded()

    def Get_Debug_File_Handle( self ):              return self.debug_log_file_handle

    def Is_Model_Loaded( self ):                    return self.model.Is_Model_Loaded()

    def Is_Model_Data_Prepared( self ):             return self.model_data_prepared

    def Get_Version( self ):                        return self.version


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################

    def Set_Data_Loader( self, new_data_loader ):   self.data_loader = new_data_loader


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Imported And Executed From A Driver Script ****" )
    exit()
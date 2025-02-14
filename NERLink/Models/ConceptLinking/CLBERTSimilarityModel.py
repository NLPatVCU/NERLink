#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/06/2022                                                                   #
#    Revised: 11/12/2022                                                                   #
#                                                                                          #
#    Generates A Concept Linking Neural Network Used For BioCreative 2021 - Task 2.        #
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
import os, re

# Suppress Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes Tensorflow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.sparse import csr_matrix
from sparse import COO

#tf.logging.set_verbosity( tf.logging.ERROR )                       # Tensorflow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # Tensorflow v1.x

if re.search( r"^2.\d+", tf.__version__ ):
    import transformers
    transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements

    import tensorflow.keras.backend as K
    from tensorflow.keras        import optimizers
    from tensorflow.keras        import regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Layer, Lambda, Reshape, TimeDistributed
    from transformers            import BertConfig, TFBertModel, TFBertForTokenClassification
else:
    import keras.backend as K
    from keras        import optimizers
    from keras        import regularizers
    from keras.models import Model
    from keras.layers import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Layer, Lambda, Reshape, TimeDistributed

# Custom Modules
from NERLink.Models.Base           import BaseModel
from NERLink.Layers                import Embedding_Extraction_Layer


############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class CLBERTSimilarityModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, margin = 30.0, optimizer = 'adam', activation_function = 'tanh',
                  loss_function = "cosine_similarity", number_of_hidden_dimensions = 200, final_layer_type = "dense",
                  embedding_dimension_size = 200, learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.1, batch_size = 32, scale = 0.35,
                  prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0",
                  verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_monitor = "val_loss",
                  early_stopping_patience = 3, use_batch_normalization = False, trainable_weights = False, embedding_a_path = "", embedding_b_path = "",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, model_path = "bert-base-cased", output_embedding_type = "average",
                  class_weights = None, sample_weights = None, use_cosine_annealing = False, cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4 ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, optimizer = optimizer, activation_function = activation_function,
                          loss_function = loss_function, number_of_hidden_dimensions = number_of_hidden_dimensions, prediction_threshold = prediction_threshold,
                          shuffle = shuffle, use_csr_format = use_csr_format, batch_size = batch_size, embedding_dimension_size = embedding_dimension_size,
                          learning_rate = learning_rate, epochs = epochs, per_epoch_saving = per_epoch_saving, use_gpu = use_gpu, momentum = momentum,
                          device_name = device_name, verbose = verbose, debug_log_file_handle = debug_log_file_handle, dropout = dropout,
                          enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                          early_stopping_monitor = early_stopping_monitor, margin = margin, final_layer_type = final_layer_type,
                          early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization,
                          trainable_weights = trainable_weights, embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path,
                          feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, model_path = model_path,
                          output_embedding_type = output_embedding_type, class_weights = class_weights, sample_weights = sample_weights,
                          use_cosine_annealing = use_cosine_annealing, cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        self.version       = 0.03
        self.network_model = "concept_linking_bert_embedding_similarity"
        self.bert_config   = None

        if self.Get_Is_Using_TF2() == False:
            self.Print_Log( "CLBERTSimilarityModel::__init__() - Error: BERT Model Only Supports Tensoflow >= 2.x", force_print = True )
            exit()


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Trains Model Using Training Data, Fits Model To Data

        Inputs:
            epochs           : Number Of Training Epochs (Integer)
            batch_size       : Size Of Each Training Batch (Integer)
            momentum         : Momentum Value (Float)
            dropout          : Dropout Value (Float)
            verbose          : Sets Training Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)
            per_epoch_saving : Toggle To Save Model After Each Training Epoch (Boolean: True, False)
            use_csr_format   : Toggle To Use Compressed Sparse Row (CSR) Formatted Matrices For Storing Training/Evaluation Data (Boolean: True, False)

        Outputs:
            None
    """
    def Fit( self, inputs = None, outputs = None, val_inputs = None, val_outputs = None, epochs = None, batch_size = None,
             verbose = None, shuffle = None, per_epoch_saving = None, class_weights = None, sample_weights = None,
             use_cosine_annealing = None, cosine_annealing_min = None, cosine_annealing_max = None ):
        # Check(s)
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 4 ):
            self.Print_Log( "CLBERTSimilarityModel::Fit() - Error: Expected Input Tuple Of 4 Elements", force_print = True )
            return False

        if outputs is None or len( outputs ) == 0:
            self.Print_Log( "CLBERTSimilarityModel::Fit() - Error: Expected Output Data Length Mismatch", force_print = True )
            return False

        # Update 'BaseModel' Class Variables
        if epochs               is not None: self.Set_Epochs( epochs )
        if batch_size           is not None: self.Set_Batch_Size( batch_size )
        if verbose              is not None: self.Set_Verbose( verbose )
        if shuffle              is not None: self.Set_Shuffle( shuffle )
        if per_epoch_saving     is not None: self.Set_Per_Epoch_Saving( per_epoch_saving )
        if class_weights        is not None: self.Set_Class_Weights( class_weights )
        if sample_weights       is not None: self.Set_Sample_Weights( sample_weights )
        if use_cosine_annealing is not None: self.Set_Use_Cosine_Annealing( use_cosine_annealing )
        if cosine_annealing_min is not None: self.Set_Cosine_Annealing_Min( cosine_annealing_min )
        if cosine_annealing_max is not None: self.Set_Cosine_Annealing_Max( cosine_annealing_max )

        # Add Model Callback Functions
        super().Add_Enabled_Model_Callbacks()

        # Extract Model Training Data From 'inputs' Parameter
        input_ids, attention_masks, token_type_ids, entry_term_masks = inputs

        # Check For Validation Data - Note: We're Not Checking The Output Data, We're Assuming It's Encoded With The Input Data
        if val_inputs is not None and val_outputs is not None:
            if isinstance( val_inputs, list ) and len( val_inputs ) == 0:
                validation_data = None
            else:
                validation_data = ( val_inputs, val_outputs )
        else:
            validation_data = None

        self.trained_instances = input_ids.shape[0]

        self.Print_Log( "CLBERTSimilarityModel::Fit() - Model Training Settings" )
        self.Print_Log( "                             - Epochs             : " + str( self.epochs             ) )
        self.Print_Log( "                             - Batch Size         : " + str( self.batch_size         ) )
        self.Print_Log( "                             - Verbose            : " + str( self.verbose            ) )
        self.Print_Log( "                             - Shuffle            : " + str( self.shuffle            ) )
        self.Print_Log( "                             - Use CSR Format     : " + str( self.use_csr_format     ) )
        self.Print_Log( "                             - Per Epoch Saving   : " + str( self.per_epoch_saving   ) )
        self.Print_Log( "                             - No. of Train Inputs: " + str( self.trained_instances  ) )

        # Perform Model Training
        self.Print_Log( "CLBERTSimilarityModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle
            if self.Get_Trainable_Weights(): self.model.trainable = True

            self.model_history = self.model.fit( [input_ids, attention_masks, token_type_ids, entry_term_masks], outputs, validation_data = validation_data,
                                                  shuffle = self.shuffle, batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose,
                                                  callbacks = self.callback_list, class_weight = self.Get_Class_Weights(), sample_weight = self.Get_Sample_Weights() )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "CLBERTSimilarityModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "CLBERTSimilarityModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "CLBERTSimilarityModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "CLBERTSimilarityModel::Fit() - Complete" )
        return True

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            inputs     : Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids', 'entry_term_masks' )
            verbose    : Sets Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)

        Outputs:
            prediction : NER Sub-Word Predictions (Numpy Array or String)
    """
    def Predict( self, inputs, verbose = 0 ):
        # Check(s)
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 4 ):
            self.Print_Log( "CLBERTSimilarityModel::Predict() - Error: Expected Input Tuple Of 4 Elements", force_print = True )
            return []

        self.Print_Log( "CLBERTSimilarityModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids, entry_term_masks = inputs

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids        = np.expand_dims( input_ids,        axis = 0 )
                attention_masks  = np.expand_dims( attention_masks,  axis = 0 )
                token_type_ids   = np.expand_dims( token_type_ids,   axis = 0 )
                entry_term_masks = np.expand_dims( entry_term_masks, axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            return self.model.predict( [input_ids, attention_masks, token_type_ids, entry_term_masks],
                                        batch_size = self.batch_size, verbose = verbose )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            inputs  : Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids', 'entry_term_masks' )

        Outputs:
            Metrics : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs, outputs, verbose, warmup_model = True ):
        # Check(s)
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 4 ):
            self.Print_Log( "CLBERTSimilarityModel::Evaluate() - Error: Expected Input Tuple Of 4 Elements", force_print = True )
            return -1, -1, -1, -1, -1

        if outputs is None or len( outputs ) == 0:
            self.Print_Log( "CLBERTSimilarityModel::Evaluate() - Error: Expected Output Data Length Mismatch", force_print = True )
            return -1, -1, -1, -1, -1

        self.Print_Log( "CLBERTSimilarityModel::Evaluate() - Executing Model Evaluation" )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids, entry_term_masks = inputs

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids        = np.expand_dims( input_ids,        axis = 0 )
                attention_masks  = np.expand_dims( attention_masks,  axis = 0 )
                token_type_ids   = np.expand_dims( token_type_ids,   axis = 0 )
                entry_term_masks = np.expand_dims( entry_term_masks, axis = 0 )

            if outputs.ndim == 1:
                outputs          = np.expand_dims( outputs,          axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            loss, accuracy, precision, recall, f1_score = self.model.evaluate( [input_ids, attention_masks, token_type_ids, entry_term_masks], outputs,
                                                                                batch_size = self.batch_size, verbose = verbose )

            self.Print_Log( "CLBERTSimilarityModel::Evaluate() - Complete" )

            return loss, accuracy, precision, recall, f1_score

    ############################################################################################
    #                                                                                          #
    #    Keras Model(s)                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Build The Keras Model

        Inputs:
            embedding_dimension_size : (Integer)
            number_of_unique_terms   : (Integer)
            number_of_outputs        : (Integer)
            final_layer_type         : Specifies Final Layer - Dense, CosFace, ArcFace, SphereFace (String)
            embeddings               : Embedding Matrix - Used For Embedding Layer (Numpy 2D Array)
            weight_decay             : Weight Decay Constant For L2 Regularizer (Float)

        Outputs:
            None
    """
    def Build_Model( self, max_sequence_length = 0, number_of_unique_terms = 0, embedding_dimension_size = 0,
                     number_of_inputs = 0, number_of_outputs = 0, embeddings = [], use_crf = False,
                     final_layer_type = None, weight_decay = 0.0001 ):
        # Update 'BaseModel' Class Variables
        if embedding_dimension_size != self.embedding_dimension_size: self.embedding_dimension_size = embedding_dimension_size
        if number_of_inputs         != self.number_of_inputs:         self.number_of_inputs         = number_of_inputs
        if number_of_outputs        != self.number_of_outputs:        self.number_of_outputs        = number_of_outputs
        if final_layer_type         is not None:                      self.final_layer_type         = final_layer_type

        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - Model Settings" )
        self.Print_Log( "                                     - Network Model             : " + str( self.network_model            ) )
        self.Print_Log( "                                     - Final Layer Type          : " + str( self.final_layer_type         ) )
        self.Print_Log( "                                     - Learning Rate             : " + str( self.learning_rate            ) )
        self.Print_Log( "                                     - Dropout                   : " + str( self.dropout                  ) )
        self.Print_Log( "                                     - Momentum                  : " + str( self.momentum                 ) )
        self.Print_Log( "                                     - Optimizer                 : " + str( self.optimizer                ) )
        self.Print_Log( "                                     - Margin                    : " + str( self.margin                   ) )
        self.Print_Log( "                                     - Scale                     : " + str( self.scale                    ) )
        self.Print_Log( "                                     - Weight Decay              : " + str( weight_decay                  ) )
        self.Print_Log( "                                     - # Of Primary Embeddings   : " + str( number_of_unique_terms        ) )
        self.Print_Log( "                                     - No. of Inputs             : " + str( self.number_of_inputs         ) )
        self.Print_Log( "                                     - Embedding Dimension Size  : " + str( self.embedding_dimension_size ) )
        self.Print_Log( "                                     - No. of Outputs            : " + str( self.number_of_outputs        ) )
        self.Print_Log( "                                     - Trainable Weights         : " + str( self.trainable_weights        ) )
        self.Print_Log( "                                     - Feature Scaling Value     : " + str( self.feature_scale_value      ) )

        # Check(s)
        final_layer_type_list = ["dense", "arcface", "cosface", "sphereface"]

        if self.final_layer_type not in final_layer_type_list:
            self.Print_Log( "CLBERTSimilarityModel::Build_Model() - Error: Invalid Final Layer Type", force_print = True )
            self.Print_Log( "                                     - Options         : " + str( final_layer_type_list ), force_print = True )
            self.Print_Log( "                                     - Specified Option: " + str( self.final_layer_type ), force_print = True )
            return False

        #######################
        #                     #
        #  Build BERT Model   #
        #                     #
        #######################

        # Setup BERT Model Configuration
        bert_model          = self.model_path

        # This Is Technically Not Needed As The BERT Configuration Is Automatically Loaded When Calling 'TFBertModel.from_pretrained()'
        self.bert_config    = BertConfig.from_pretrained( bert_model, num_labels = number_of_outputs )

        # Set Embedding Dimension Size
        embedding_dimension_size = self.bert_config.hidden_size
        if self.embedding_dimension_size != embedding_dimension_size: self.embedding_dimension_size = embedding_dimension_size

        # Setup The BERT Model - Determine If We're Loading From A File Or HuggingFace Model Archive By Model Name
        load_from_file      = True if ".bin" in bert_model or "./" in bert_model else False
        Encoder             = TFBertModel.from_pretrained( bert_model, from_pt = load_from_file, config = self.bert_config )
        self.model_encoder_layer = Encoder.bert

        # Determine If We're Refining BERT Layers In Addition To The Attached Layers Or Just Training The Attached Layers
        #   i.e. Set Encoder Layer Weights/Variables To Trainable Or Freeze Them
        Encoder.trainable   = self.trainable_weights

        # BERT Model Inputs
        token_input_ids     = Input( shape = ( max_sequence_length, ), name = "Token_ID_Input",        dtype = tf.int32 )
        attention_mask      = Input( shape = ( max_sequence_length, ), name = "Attention_Mask_Input",  dtype = tf.int32 )
        token_type_ids      = Input( shape = ( max_sequence_length, ), name = "Token_Type_ID_Input",   dtype = tf.int32 )
        entry_term_mask     = Input( shape = ( max_sequence_length, ), name = "Entry_Term_Mask_Input", dtype = tf.int32 )

        # Pass BERT-Specific Inputs To The Encoder, Extract Embeddings Per Input Sequence Sub-Word
        #   NOTE: Calling "Encoder( input_ids, ... )" Achieves The Same, But Results In A Nested Layer Error When Saving, Then Loading The Model.
        #         Calling "Encoder.bert( inputs_ids, ... )" Resolves The Nested Layer Issue.
        #         Source: https://github.com/keras-team/keras/issues/14345
        encoder_inputs      = Encoder.bert( input_ids = token_input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids )[0]   # Same as 'embedding.last_hidden_state' In TFBertModel

        # Transform 'entry_term_mask' To Match The Shape Of BERT Output Token/Sub-Word Embeddings: ( batch_size, sequence_length, hidden_size )
        entry_term_mask_seq = Reshape( target_shape = ( entry_term_mask.shape[-1], 1 ), input_shape = ( entry_term_mask.shape[-1], ) )( entry_term_mask )

        # Compute Average Of Entry Term Sub-Words / Zero Out Other Embeddings Representations
        #   This Should Zero Out Non-Entry Term Sub-Words Only Leaving The Average Of The Sub-Word
        #   Embeddings As A Single Representation. (This Is Done For Each Instance Within The Batch.)
        sub_word_embeddings = Embedding_Extraction_Layer( output_embedding_type = self.output_embedding_type, hidden_size = embedding_dimension_size )( [encoder_inputs, entry_term_mask_seq] )

        # Feature Scale The Embedding Representation
        if self.feature_scale_value != 1.0:
            feature_scale_value = self.feature_scale_value  # Fixes Python Recursion Limit Error (Model Tries To Save All 'self' Variable When Used With Lambda Function)
            sub_word_embeddings = Lambda( lambda x: x * feature_scale_value )( sub_word_embeddings )

        if self.use_batch_normalization:
            dense_layer      = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_1' )( sub_word_embeddings )
            batch_norm_layer = BatchNormalization( name = "Batch_Norm_Layer_1" )( dense_layer )
            dropout_layer    = Dropout( name = "Dropout_Layer_1", rate = self.dropout )( batch_norm_layer )
            dense_layer      = Dense( units = number_of_outputs, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_2',
                                      kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2( weight_decay ) )( dropout_layer )
            dense_layer      = BatchNormalization( name = "Batch_Norm_Layer_2" )( dense_layer )
        else:
            dense_layer      = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_1' )( sub_word_embeddings )
            dropout_layer    = Dropout( name = "Dropout_Layer_1", rate = self.dropout )( dense_layer )
            dense_layer      = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_2',
                                      kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2( weight_decay ) )( dropout_layer )

        output_layer = Dense( units = number_of_outputs, input_dim = embedding_dimension_size, activation = self.activation_function, name = "Concept_Embedding_Output" )( dense_layer )

        self.model = Model( inputs = [token_input_ids, attention_mask, token_type_ids, entry_term_mask], outputs = output_layer, name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( learning_rate = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( learning_rate = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd )

        # Print Model Summary
        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =========================================================" )
        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x: self.Print_Log( "CLBERTSimilarityModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =========================================================" )
        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =                                                       =" )
        self.Print_Log( "CLBERTSimilarityModel::Build_Model() - =========================================================" )

        return True


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
    print( "     from models import CLBERTSimilarityModel\n" )
    print( "     model = CLBERTSimilarityModel( print_debug_log = True, per_epoch_saving = False, use_csr_format = True )" )
    print( "     model.Fit( \"data/cui_mini\", epochs = 30, batch_size = 4, verbose = 1 )" )
    exit()

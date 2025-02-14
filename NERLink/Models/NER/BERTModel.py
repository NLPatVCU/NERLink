#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    06/24/2021                                                                   #
#    Revised: 11/12/2022                                                                   #
#                                                                                          #
#    Generates A BERT-Based Neural Network Used For BioCreative 2021 - Task 2.             #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Aidan Myers   - myersas@vcu.edu                                                       #
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
    import tensorflow_addons        as tfa
    from tensorflow.keras         import optimizers
    from tensorflow.keras.models  import Model
    from tensorflow.keras.layers  import Activation, BatchNormalization, Concatenate, Dense, Dot, Dropout, Embedding, Input, Lambda, Layer
    from tensorflow.keras.losses  import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
    from transformers             import BertConfig, TFBertModel, TFBertForTokenClassification
else:
    import keras.backend as K
    from keras        import optimizers
    from keras.models import Model
    from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Dot, Dropout, Embedding, Input, Lambda, Layer

# Custom Modules
from NERLink.Models.Base           import BaseModel


############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class BERTModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, optimizer = 'adam', activation_function = 'relu',
                  loss_function = "sparse_categorical_crossentropy", number_of_embedding_dimensions = 200, bilstm_dimension_size = 64,
                  bilstm_merge_mode = "concat", learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.1,
                  batch_size = 32, prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True,
                  device_name = "/gpu:0", verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False,
                  early_stopping_monitor = "val_loss", early_stopping_patience = 3, use_batch_normalization = False, trainable_weights = False,
                  embedding_a_path = "", embedding_b_path = "", final_layer_type = "dense", feature_scale_value = 1.0, learning_rate_decay = 0.004,
                  model_path = "bert-base-cased", class_weights = None, sample_weights = None, use_cosine_annealing = False, cosine_annealing_min = 1e-6,
                  cosine_annealing_max = 2e-4 ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, momentum = momentum, bilstm_merge_mode = bilstm_merge_mode,
                          optimizer = optimizer, activation_function = activation_function, batch_size = batch_size, prediction_threshold = prediction_threshold,
                          shuffle = shuffle, use_csr_format = use_csr_format, loss_function = loss_function, embedding_dimension_size = number_of_embedding_dimensions,
                          learning_rate = learning_rate, bilstm_dimension_size = bilstm_dimension_size, epochs = epochs, dropout = dropout, per_epoch_saving = per_epoch_saving,
                          use_gpu = use_gpu, device_name = device_name, verbose = verbose, debug_log_file_handle = debug_log_file_handle,
                          enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, final_layer_type = final_layer_type,
                          early_stopping_monitor = early_stopping_monitor, early_stopping_patience = early_stopping_patience,
                          use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_a_path = embedding_a_path,
                          embedding_b_path = embedding_b_path, feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay,
                          model_path = model_path, class_weights = class_weights, sample_weights = sample_weights, use_cosine_annealing = use_cosine_annealing,
                          cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        self.version        = 0.08
        self.network_model  = "ner_bert"   # Force Setting Model To 'BERT' Model.
        self.bert_config    = None
        self.sparse_ce_loss = SparseCategoricalCrossentropy( from_logits = True )   # Used For Debugging Purposes

        if self.Get_Is_Using_TF2() == False:
            self.Print_Log( "BERTModel::__init__() - Error: BERT Model Only Supports Tensoflow >= 2.x", force_print = True )
            exit()


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Saves Model To File

        Inputs:
            file_path   : File Path (String)
            save_format : Model Save Format - Defaults: 'tf' -> Tensorflow 2.x / 'h5' -> Tensorflow 1.x
            save_config : Saves Model Configuration

        Outputs:
            None
    """
    def Save_Model( self, model_path, save_format = "h5", save_config = False ):
        # BERT Model Saves In TF2.x And Is Unable To Save The Model Configuration
        super().Save_Model( model_path = model_path, save_format = save_format, save_config = False )

    """
        Trains Model Using Training Data, Fits Model To Data

        Inputs:
            inputs           : Model Inputs  - Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids' )
            outputs          : Model OUtputs - Three Dimensional Numpy Array ( number_of_instances, sequence_length, number_of_output_labels )
            epochs           : Number Of Training Epochs (Integer)
            batch_size       : Size Of Each Training Batch (Integer)
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
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 3 ):
            self.Print_Log( "BERTModel::Fit() - Error: Expected Input Tuple Of 3 Elements", force_print = True )
            return False

        if outputs is None or len( outputs ) == 0:
            self.Print_Log( "BERTModel::Fit() - Error: Expected Output Data Length Mismatch", force_print = True )
            return False

        # Class Weight Check
        if class_weights:
            self.Print_Log( "BERTModel::Fit() - Warning: Class Weights Not Supported / Setting 'class_weights = None'", force_print = True )
            self.class_weights = None

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
        input_ids, attention_masks, token_type_ids = inputs

        # Check For Validation Data - Note: We're Not Checking The Output Data, We're Assuming It's Encoded With The Input Data
        if val_inputs is not None and val_outputs is not None:
            if isinstance( val_inputs, list       ) and len( val_inputs )   == 0 or \
               isinstance( val_inputs, COO        ) and val_inputs.shape[0] == 0 or \
               isinstance( val_inputs, csr_matrix ) and val_inputs.shape[0] == 0:
                validation_data = None
            else:
                validation_data = ( val_inputs, val_outputs )
        else:
            validation_data = None

        self.trained_instances = input_ids.shape[0]

        self.Print_Log( "BERTModel::Fit() - Model Training Settings" )
        self.Print_Log( "                 - Epochs             : " + str( self.epochs            ) )
        self.Print_Log( "                 - Batch Size         : " + str( self.batch_size        ) )
        self.Print_Log( "                 - Verbose            : " + str( self.verbose           ) )
        self.Print_Log( "                 - Shuffle            : " + str( self.shuffle           ) )
        self.Print_Log( "                 - Use CSR Format     : " + str( self.use_csr_format    ) )
        self.Print_Log( "                 - Per Epoch Saving   : " + str( self.per_epoch_saving  ) )
        self.Print_Log( "                 - No. of Instances   : " + str( self.trained_instances ) )

        # Data Generator Placeholders
        data_generator, validation_generator = None, None

        # Compute Number Of Steps Per Batch (Use CSR Format == True)
        steps_per_batch, val_steps_per_batch = 0, None

        # Compute Model Data Steps Per Batch Value
        if self.batch_size >= self.trained_instances:
            steps_per_batch = 1
        else:
            steps_per_batch = self.trained_instances // self.batch_size if self.trained_instances % self.batch_size == 0 else self.trained_instances // self.batch_size + 1

        # Compute Model Validation Data Steps Per Batch Value
        if validation_data:
            validation_instances = validation_data[0][0].shape[0]

            if self.batch_size >= validation_instances:
                val_steps_per_batch = 1
            else:
                val_steps_per_batch = validation_instances // self.batch_size if validation_instances % self.batch_size == 0 else validation_instances // self.batch_size + 1

        # Perform Model Training
        self.Print_Log( "BERTModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle
            if self.Get_Trainable_Weights(): self.model.trainable = True

            self.model_history = self.model.fit( [input_ids, attention_masks, token_type_ids], outputs, validation_data = validation_data,
                                                  batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose,
                                                  callbacks = self.callback_list, class_weight = self.Get_Class_Weights(), sample_weight = self.Get_Sample_Weights() )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "BERTModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "BERTModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "BERTModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "BERTModel::Fit() - Complete" )
        return True

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            inputs     : Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids' )
            verbose    : Sets Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)

        Outputs:
            prediction : NER Sub-Word Predictions (Numpy Array or String)
    """
    def Predict( self, inputs, verbose = 0 ):
        # Check(s)
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 3 ):
            self.Print_Log( "BERTModel::Predict() - Error: Expected Input Tuple Of 3 Elements", force_print = True )
            return []

        self.Print_Log( "BERTModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids = inputs

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids       = np.expand_dims( input_ids,       axis = 0 )
                attention_masks = np.expand_dims( attention_masks, axis = 0 )
                token_type_ids  = np.expand_dims( token_type_ids,  axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            return self.model.predict( [input_ids, attention_masks, token_type_ids], batch_size = self.batch_size, verbose = verbose )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            inputs  : Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids' )

        Outputs:
            Metrics : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs, outputs, verbose, warmup_model = True ):
        # Check(s)
        if inputs is None or ( isinstance( inputs, tuple ) and len( inputs ) < 3 ):
            self.Print_Log( "BERTModel::Evaluate() - Error: Expected Input Tuple Of 3 Elements", force_print = True )
            return -1, -1, -1, -1, -1

        if outputs is None or len( outputs ) == 0:
            self.Print_Log( "BERTModel::Evaluate() - Error: Expected Output Data Length Mismatch", force_print = True )
            return -1, -1, -1, -1, -1

        self.Print_Log( "BERTModel::Evaluate() - Executing Model Evaluation" )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids = inputs

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids       = np.expand_dims( input_ids,       axis = 0 )
                attention_masks = np.expand_dims( attention_masks, axis = 0 )
                token_type_ids  = np.expand_dims( token_type_ids,  axis = 0 )

            if outputs.ndim == 1:
                outputs         = np.expand_dims( outputs,         axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            loss, accuracy  = self.model.evaluate( [input_ids, attention_masks, token_type_ids], outputs, verbose = verbose )

            # Following Metrics Not Implemented In Model
            #   Report -1 Scores To Maintain Expected 5 Packed Metric Values
            precision, recall, f1_score = -1, -1, -1

            self.Print_Log( "BERTModel::Evaluate() - Complete" )

            return loss, accuracy, precision, recall, f1_score

    ############################################################################################
    #                                                                                          #
    #    Keras Model Loss                                                                      #
    #                                                                                          #
    ############################################################################################

    # Used For Debugging Purposes
    def Debug_Sparse_Categorical_Crossentropy_Loss( self, y_true, y_pred ):
        # Print Model Predicted Indices Per Token And True Label Indices
        pred_idx = tf.cast( tf.argmax( y_pred, axis = -1 ), tf.int32 )
        tf.print( "\ny_pred:", pred_idx, summarize = -1 )
        tf.print( "y_true:", y_true, summarize = -1 )

        return self.sparse_ce_loss( y_true, y_pred )

    # Computes Sparse Categorical Crossentropy Against Actual Sub-Word Token Predicted & True Labels
    #   Ignores Masked Sub-Word Labels
    def Masked_Sparse_Categorical_Crossentropy( y_true, y_pred ):
        mask_value    = -100
        y_true_masked = tf.boolean_mask( y_true, tf.not_equal( y_true, mask_value ) )
        y_pred_masked = tf.boolean_mask( y_pred, tf.not_equal( y_true, mask_value ) )
        return tf.reduce_mean( sparse_categorical_crossentropy( y_true_masked, y_pred_masked ) )

    ############################################################################################
    #                                                                                          #
    #    Keras Model(s)                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Build The Keras Model

        Inputs:
            max_sequence_length      : (Integer)
            number_of_unique_terms   : (Integer) (Not Used)
            embedding_dimension_size : (Integer) (Not Used)
            number_of_outputs        : (Integer)
            embeddings               : Numpy 2-Dimensional Array (Not Used)

        Outputs:
            None
    """
    def Build_Model( self, max_sequence_length = 0, number_of_unique_terms = 0, embedding_dimension_size = 0,
                     number_of_inputs = 0, number_of_outputs = 0, embeddings = [], use_crf = False ):
        # Update 'BaseModel' Class Variables
        if number_of_inputs  != self.number_of_inputs:  self.number_of_inputs  = number_of_inputs
        if number_of_outputs != self.number_of_outputs: self.number_of_outputs = number_of_outputs

        self.Print_Log( "BERTModel::Build_Model() - Model Settings" )
        self.Print_Log( "                         - Network Model              : " + str( self.network_model            ) )
        self.Print_Log( "                         - Learning Rate              : " + str( self.learning_rate            ) )
        self.Print_Log( "                         - Dropout                    : " + str( self.dropout                  ) )
        self.Print_Log( "                         - Use CRF Layer              : " + str( use_crf                       ) )
        self.Print_Log( "                         - Momentum                   : " + str( self.momentum                 ) )
        self.Print_Log( "                         - Optimizer                  : " + str( self.optimizer                ) )
        self.Print_Log( "                         - Activation Function        : " + str( self.activation_function      ) )
        self.Print_Log( "                         - No. of Unique Terms        : " + str( self.number_of_features       ) )
        self.Print_Log( "                         - Embedding Dimension Size   : " + str( self.embedding_dimension_size ) )
        self.Print_Log( "                         - No. of Outputs             : " + str( self.number_of_outputs        ) )
        self.Print_Log( "                         - Trainable Weights          : " + str( self.trainable_weights        ) )
        self.Print_Log( "                         - Feature Scaling Value      : " + str( self.feature_scale_value      ) )
        self.Print_Log( "                         - Max Sequence Length        : " + str( max_sequence_length           ) )

        # Metrics
        accuracy = "accuracy"

        # Check We're Using The Appropriate Loss Function
        if self.loss_function != "sparse_categorical_crossentropy":
            self.Print_Log( "BERTModel::Build_Model() - Warning: Only Sparse Categorical Crossentropy Loss Function Is Supported", force_print = True )
            self.loss_function = "sparse_categorical_crossentropy"

        #######################
        #                     #
        #  Build BERT Model   #
        #                     #
        #######################

        # Setup BERT Model Configuration
        bert_model        = self.model_path

        # This Is Technically Not Needed As The BERT Configuration Is Automatically Loaded When Calling 'TFBertModel.from_pretrained()'
        self.bert_config  = BertConfig.from_pretrained( bert_model, num_labels = number_of_outputs )

        # Setup The BERT Model - Determine If We're Loading From A File Or HuggingFace Model Archive By Model Name
        load_from_file    = True if ".bin" in bert_model or "./" in bert_model else False
        Encoder           = TFBertModel.from_pretrained( bert_model, from_pt = load_from_file, config = self.bert_config )
        self.model_encoder_layer = Encoder.bert

        # Determine If We're Refining BERT Layers In Addition To The Attached Layers Or Just Training The Attached Layers
        #   i.e. Set Encoder Layer Weights/Variables To Trainable Or Freeze Them
        Encoder.trainable = self.trainable_weights

        # BERT NER Model
        token_input_ids   = Input( shape = ( max_sequence_length, ), dtype = tf.int32 )
        attention_mask    = Input( shape = ( max_sequence_length, ), dtype = tf.int32 )
        token_type_ids    = Input( shape = ( max_sequence_length, ), dtype = tf.int32 )
        embedding         = Encoder.bert( input_ids = token_input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids )[0]   # Same as 'embedding.last_hidden_state' In TFBertModel
        embedding         = Dropout( name = "Dropout_Layer_1", rate = self.dropout )( embedding )
        output            = Dense( units = number_of_outputs, activation = self.activation_function )( embedding )

        # ToDo: Complete / Implement Loss Function - Enabling 'use_crf = True' Will Result In An Error
        #       Warning: Do Not Use
        #       The Loss Function Is Not Implemented (Log-Likelihood Loss)
        #
        #       CRF Outputs: Index 0 - Sequence
        #                    Index 1 - Potentials
        #                    index 2 - Sequence Length
        #
        if use_crf:
            crf           = tfa.layers.CRF( units = number_of_outputs )
            output        = crf( output )[1]

        self.model        = Model( inputs = [token_input_ids, attention_mask, token_type_ids], outputs = output, name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( learning_rate = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt,
                                sample_weight_mode = "temporal", metrics = [ accuracy ] )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( learning_rate = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd,
                                sample_weight_mode = "temporal", metrics = [ accuracy ] )

        # Print Model Summary
        self.Print_Log( "BERTModel::Build_Model() - =========================================================" )
        self.Print_Log( "BERTModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "BERTModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x: self.Print_Log( "BERTModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "BERTModel::Build_Model() - =========================================================" )
        self.Print_Log( "BERTModel::Build_Model() - =                                                       =" )
        self.Print_Log( "BERTModel::Build_Model() - =========================================================" )

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
    exit()

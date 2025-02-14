#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/12/2022                                                                   #
#    Revised: 11/19/2022                                                                   #
#                                                                                          #
#    Generates A NER + CL Multi-Task Model Intended For The                                #
#          BioCreative VII - Track II Task.                                                #
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
    from tensorflow.keras         import optimizers
    from tensorflow.keras         import regularizers
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.keras.models  import Model
    from tensorflow.keras.layers  import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Layer, Lambda, Reshape, TimeDistributed
    from tensorflow.keras.losses  import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
    from transformers             import BertConfig, TFBertModel, TFBertForTokenClassification
else:
    import keras.backend as K
    from keras                    import optimizers
    from keras                    import regularizers
    from keras.models             import Model
    from keras.layers             import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Layer, Lambda, Reshape, TimeDistributed

# Custom Modules
from NERLink.DataGenerator         import DataGenerator
from NERLink.Layers                import ArcFace, CosFace, SphereFace
from NERLink.Models.Base           import BaseModel


############################################################################################
#                                                                                          #
#    Custom DataGenerator Class                                                            #
#                                                                                          #
############################################################################################

class ModelDataGenerator( DataGenerator ):
    def __init__( self, X, Y, batch_size = 1, number_of_instances = None, shuffle = True, sample_weights = None ):
        super().__init__( X = X, Y = Y, batch_size = batch_size, number_of_instances = number_of_instances,
                          shuffle = shuffle, sample_weights = sample_weights )
    'Generate One Batch Of Data'
    def __getitem__( self, index ):
        # Generate Batch Given Start And End Indices
        start_index   = self.batch_size * index
        end_index     = self.batch_size * ( index + 1 )
        batch_indices = self.indices[start_index:end_index]

        # Extract Indices Given Inputs
        token_input_ids, attention_masks, token_type_ids = self.X
        ner_output, cl_output                            = self.Y
        X_token_input_ids  = token_input_ids[batch_indices,:]
        X_attention_mask   = attention_masks[batch_indices,:]
        X_token_type_ids   = token_type_ids[batch_indices,:]
        Y_ner_output       = ner_output[batch_indices,:].todense()
        Y_cl_output        = cl_output[batch_indices,:].todense()

        if self.sample_weights is not None and len( self.sample_weights ) > 1:
            if isinstance( self.sample_weights, COO ) or isinstance( self.sample_weights, csr_matrix ):
                ner_sample_weights = self.sample_weights[0][batch_indices,:].todense()
                cl_sample_weights  = self.sample_weights[1][batch_indices,:].todense()
                sample_weights     = [ ner_sample_weights, cl_sample_weights ]
            else:
                ner_sample_weights = self.sample_weights[0][batch_indices,:]
                cl_sample_weights  = self.sample_weights[1][batch_indices,:]
                sample_weights     = [ ner_sample_weights, cl_sample_weights ]
        else:
            sample_weights = None

        return [X_token_input_ids, X_attention_mask, X_token_type_ids], [Y_ner_output, Y_cl_output], sample_weights


############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class NERCLMultiTaskModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, margin = 30.0, optimizer = 'adam', activation_function = 'softmax',
                  loss_function = "categorical_crossentropy", number_of_hidden_dimensions = 200, final_layer_type = "dense",
                  embedding_dimension_size = 200, learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.1, batch_size = 32, scale = 0.35,
                  prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0",
                  verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_monitor = "val_loss",
                  early_stopping_patience = 3, use_batch_normalization = False, trainable_weights = False, embedding_a_path = "", embedding_b_path = "",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, model_path = "bert-base-cased", output_embedding_type = "average", class_weights = None,
                  sample_weights = None, use_cosine_annealing = False, cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4 ):
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
        self.version        = 0.03
        self.network_model  = "ner_concept_linking_multi_task_bert"
        self.bert_config    = None
        self.sparse_ce_loss = SparseCategoricalCrossentropy( from_logits = True, reduction = tf.keras.losses.Reduction.NONE )
        self.debug_sce_loss = SparseCategoricalCrossentropy( from_logits = True )   # Used For Debugging Purposes

        if self.Get_Is_Using_TF2() == False:
            self.Print_Log( "NERCLMultiTaskModel::__init__() - Error: BERT Model Only Supports Tensoflow >= 2.x", force_print = True )
            exit()


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Converts Randomized Batches Of Model Inputs & Outputs From CSR_Matrix Format
          To Numpy Arrays For Model Training

        Inputs:
            X               : List Of Arrays (Token Input IDs, Attention Masks, Token Type IDs & Entry Term Masks)
            Y               : Model Outputs (CSR_Matrix)
            batch_size      : Batch Size (Integer)
            steps_per_batch : Number Of Iterations Per Epoch (Integer)
            shuffle         : Shuffles Data Prior To Conversion (Boolean)

        Outputs:
            token_input_ids : Numpy 2D Matrix Of Token Input IDs (Numpy Array)
            attention_mask  : Numpy 2D Matrix Of Attention Masks (Numpy Array)
            token_type_ids  : Numpy 2D Matrix Of Token Type IDs (Numpy Array)
            entry_term_mask : Numpy 2D Matrix Of Entry Term Masks (Numpy Array)
            Y_output        : Numpy 2D Matrix Of Model Outputs (Numpy Array)

            Modification Of Code From Source: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
    """
    def Batch_Generator( self, X, Y, batch_size, steps_per_batch, shuffle ):
        token_input_ids, attention_masks, token_type_ids = X
        ner_output, cl_output = Y
        number_of_instances   = token_input_ids.shape[0]      # Should Be The Same As 'self.trained_instances'
        counter               = 0
        sample_index          = np.arange( number_of_instances )

        if shuffle:
            np.random.shuffle( sample_index )

        while True:
            start_index = batch_size * counter
            end_index   = batch_size * ( counter + 1 )

            # Check - Fixes Batch_Generator Training Errors With The Number Of Instances % Batch Sizes != 0
            end_index   = number_of_instances if end_index > number_of_instances else end_index

            batch_index       = sample_index[start_index:end_index]
            X_token_input_ids = token_input_ids[batch_index,:]
            X_attention_mask  = attention_masks[batch_index,:]
            X_token_type_ids  = token_type_ids[batch_index,:]
            Y_ner_output      = ner_output[batch_index,:].todense()
            Y_cl_output       = cl_output[batch_index,:].todense()
            counter           += 1

            yield [X_token_input_ids, X_attention_mask, X_token_type_ids], [Y_ner_output, Y_cl_output]

            # Reset The Batch Index After Final Batch Has Been Reached
            if counter == steps_per_batch:
                if shuffle:
                    np.random.shuffle( sample_index )
                counter = 0

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
        if isinstance( inputs, tuple ) and len( inputs ) < 4:
            self.Print_Log( "NERCLMultiTaskModel::Fit() - Error: Expected Input Tuple Of 4 Elements", force_print = True )
            return False

        if isinstance( outputs, tuple ) and len( outputs ) < 2:
            self.Print_Log( "NERCLMultiTaskModel::Fit() - Error: Expected Output Data Length Mismatch", force_print = True )
            return False

        # Class Weighting Check
        if class_weights:
            self.Print_Log( "NERCLMultiTaskModel::Fit() - Warning: Class Weights Not Supported / Setting 'class_weights = None'", force_print = True )
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
        input_ids, attention_masks, token_type_ids, entry_term_masks = inputs

        # Check For Validation Data - Note: We're Not Checking The Output Data, We're Assuming It's Encoded With The Input Data
        if val_inputs is not None and val_outputs is not None:
            if isinstance( val_inputs, list       ) and len( val_inputs )   == 0 or \
               isinstance( val_inputs, COO        ) and val_inputs.shape[0] == 0 or \
               isinstance( val_inputs, csr_matrix ) and val_inputs.shape[0] == 0:
                validation_data = None
            else:
                # Remove 'Entry Term Masks' From Validation Inputs
                val_input_ids, val_attention_masks, val_token_type_ids, val_entry_term_masks = val_inputs
                val_inputs = ( val_input_ids, val_attention_masks, val_token_type_ids )
                validation_data = ( val_inputs, val_outputs )
        else:
            validation_data = None

        self.trained_instances = input_ids.shape[0]

        self.Print_Log( "NERCLMultiTaskModel::Fit() - Model Training Settings" )
        self.Print_Log( "                       - Epochs             : " + str( self.epochs             ) )
        self.Print_Log( "                       - Batch Size         : " + str( self.batch_size         ) )
        self.Print_Log( "                       - Verbose            : " + str( self.verbose            ) )
        self.Print_Log( "                       - Shuffle            : " + str( self.shuffle            ) )
        self.Print_Log( "                       - Use CSR Format     : " + str( self.use_csr_format     ) )
        self.Print_Log( "                       - Per Epoch Saving   : " + str( self.per_epoch_saving   ) )
        self.Print_Log( "                       - No. of Train Inputs: " + str( self.trained_instances  ) )

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
        self.Print_Log( "NERCLMultiTaskModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle
            if self.Get_Trainable_Weights(): self.model.trainable = True

            # This Is Depreciated In Tensorflow Versions Greater Than 2.x
            if self.Get_Is_Using_TF1() and self.use_csr_format:
                # Prepare Model Training Data Generator
                data_generator = self.Batch_Generator( [input_ids, attention_masks, token_type_ids], outputs, batch_size = self.batch_size,
                                                       steps_per_batch = steps_per_batch, shuffle = self.shuffle )
                # Prepare Model Validation Data Generator
                if validation_data:
                    validation_generator = self.Batch_Generator( val_inputs, val_outputs, batch_size = self.batch_size,
                                                                 steps_per_batch = val_steps_per_batch, shuffle = False )

                self.model_history = self.model.fit_generator( generator = data_generator, validation_data = validation_generator, epochs = self.epochs,
                                                               steps_per_epoch = steps_per_batch, validation_steps = val_steps_per_batch, verbose = self.verbose,
                                                               callbacks = self.callback_list, class_weight = self.class_weights )
            elif self.use_csr_format:
                data_generator = ModelDataGenerator( X = [input_ids, attention_masks, token_type_ids], Y = outputs, batch_size = self.batch_size, number_of_instances = self.trained_instances,
                                                     shuffle = self.shuffle, sample_weights = self.Get_Sample_Weights() )

                if validation_data:
                    validation_instances = validation_data[0][0].shape[0]
                    validation_generator = ModelDataGenerator( X = val_inputs, Y = val_outputs, batch_size = self.batch_size, number_of_instances = validation_instances, shuffle = self.shuffle )

                self.model_history = self.model.fit( data_generator, validation_data = validation_generator,
                                                     shuffle = self.shuffle, batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose,
                                                     callbacks = self.callback_list, class_weight = self.Get_Class_Weights() )
            else:
                self.model_history = self.model.fit( [input_ids, attention_masks, token_type_ids], outputs, validation_data = validation_data,
                                                     shuffle = self.shuffle, batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose,
                                                     callbacks = self.callback_list, class_weight = self.Get_Class_Weights(), sample_weight = self.Get_Sample_Weights() )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "NERCLMultiTaskModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "NERCLMultiTaskModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "NERCLMultiTaskModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "NERCLMultiTaskModel::Fit() - Complete" )
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
        if isinstance( inputs, tuple ) and len( inputs ) < 4:
            self.Print_Log( "NERCLMultiTaskModel::Predict() - Error: Expected Input Tuple Of 3 Elements", force_print = True )
            return None

        self.Print_Log( "NERCLMultiTaskModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids, entry_term_masks = inputs

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids        = np.expand_dims( input_ids,        axis = 0 )
                attention_masks  = np.expand_dims( attention_masks,  axis = 0 )
                token_type_ids   = np.expand_dims( token_type_ids,   axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            return self.model.predict( [input_ids, attention_masks, token_type_ids], batch_size = self.batch_size, verbose = verbose )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            inputs  : Tuple Of Numpy Arrays ( 'input_ids', 'attention_masks', 'token_type_ids', 'entry_term_masks' )

        Outputs:
            Metrics : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs, outputs, verbose, warmup_model = True ):
        # Check(s)
        if isinstance( inputs, tuple ) and len( inputs ) < 4:
            self.Print_Log( "NERCLMultiTaskModel::Evaluate() - Error: Expected Input Tuple Of 3 Elements", force_print = True )
            return -1, -1, -1, -1, -1

        if isinstance( outputs, tuple ) and len( outputs ) < 2:
            self.Print_Log( "NERCLMultiTaskModel::Evaluate() - Error: Expected Output Tuple Of 2 Elements", force_print = True )
            return -1, -1, -1, -1, -1

        self.Print_Log( "NERCLMultiTaskModel::Evaluate() - Executing Model Evaluation" )

        with tf.device( self.device_name ):
            input_ids, attention_masks, token_type_ids, entry_term_masks = inputs
            ner_labels, cl_labels = outputs

            input_ids       = np.asarray( input_ids )
            attention_masks = np.asarray( attention_masks )
            token_type_ids  = np.asarray( token_type_ids )
            ner_labels      = np.asarray( ner_labels )
            cl_labels       = np.asarray( cl_labels )

            # Ensure That Predicting For A Single Instance Retains The Correct Dimensions
            if input_ids.ndim == 1:
                input_ids        = np.expand_dims( input_ids,        axis = 0 )
                attention_masks  = np.expand_dims( attention_masks,  axis = 0 )
                token_type_ids   = np.expand_dims( token_type_ids,   axis = 0 )

            if ner_labels.ndim == 1:
                ner_labels       = np.expand_dims( ner_labels,       axis = 0 )
                cl_labels        = np.expand_dims( cl_labels,        axis = 0 )

            # Set Encoder Trainable Parameter Based On User-Specified Boolean Toggle / Freeze Encoder Weights For Inference
            if self.Get_Trainable_Weights(): self.model.trainable = False

            metrics = self.model.evaluate( [input_ids, attention_masks, token_type_ids],
                                           [ner_labels, cl_labels],
                                           batch_size = self.batch_size, verbose = verbose )

            loss, ner_loss, cl_loss, ner_accuracy, cl_accuracy = metrics
            accuracy = ( ner_accuracy + cl_accuracy ) / 2

            self.Print_Log( "NERCLMultiTaskModel::Evaluate() - Complete" )

            # Following Metrics Not Implemented In Model
            #   Report -1 Scores To Maintain Expected 5 Packed Metric Values
            precision, recall, f1_score = -1, -1, -1

            return loss, accuracy, precision, recall, f1_score

    ############################################################################################
    #                                                                                          #
    #    Keras Model Loss / Metrics                                                            #
    #                                                                                          #
    ############################################################################################

    # Used For Debugging Purposes
    def Debug_Sparse_Categorical_Crossentropy_Loss( self, y_true, y_pred ):
        # Print Model Predicted Indices Per Token And True Label Indices
        pred_idx = tf.cast( tf.argmax( y_pred, axis = -1 ), tf.int32 )
        tf.print( "\ny_pred:", pred_idx, summarize = -1 )
        tf.print( "y_true:", y_true, summarize = -1 )

        return self.debug_sce_loss( y_true, y_pred )

    # Computes Sparse Categorical Crossentropy Against Actual Sub-Word Token Predicted & True Labels
    #   Ignores Masked Sub-Word Labels
    def Masked_Sparse_Categorical_Crossentropy( self, y_true, y_pred ):
        mask_value    = -100

        # make sure only labels that are not equal to -100
        # are taken into account as loss
        if tf.math.reduce_any( y_true == -1 ):
            warnings.warn( "Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead." )
            active_loss = tf.reshape( y_true, ( -1, ) ) != -1
        else:
            active_loss = tf.reshape( y_true, ( -1, ) ) != -100

        reduced_y_pred = tf.boolean_mask( tf.reshape( y_pred, ( -1, tf.shape( y_pred )[2] ) ), active_loss )
        y_true         = tf.boolean_mask( tf.reshape( y_true, ( -1, ) ), active_loss )

        # Print Model Predictions Vs True Labels - Used For Debugging
        # y_pred_idx    = tf.cast( tf.argmax( y_pred_masked, axis = -1 ), tf.int32 )
        # tf.print( "\ny_pred:", y_pred_idx, summarize = -1 )
        # tf.print( "y_true:", y_true_masked, summarize = -1 )

        return self.sparse_ce_loss( y_true, reduced_y_pred )

    # def Masked_Sparse_Categorical_Crossentropy_Accuracy( self, y_true, y_pred ):
    #     mask_value     = -100
    #     active_loss    = tf.reshape( y_true, ( -1, ) ) != mask_value
    #     reduced_logits = tf.boolean_mask( tf.reshape( y_pred, ( -1, shape_list( y_pred )[2] ) ), active_loss )
    #     y_true         = tf.boolean_mask( tf.reshape( y_true, ( -1, ) ), active_loss )
    #     reduced_logits = tf.cast( tf.argmax( reduced_logits, axis = -1 ), tf.keras.backend.floatx() )
    #     equality       = tf.equal( y_true, reduced_logits )
    #     return tf.reduce_mean( tf.cast( equality, tf.keras.backend.floatx() ) )

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
            number_of_inputs         : (Integer)
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
        if number_of_inputs         != self.number_of_inputs :        self.number_of_inputs         = number_of_inputs
        if number_of_outputs        != self.number_of_outputs:        self.number_of_outputs        = number_of_outputs
        if final_layer_type         is not None:                      self.final_layer_type         = final_layer_type

        number_of_ner_outputs, number_of_cl_outputs = number_of_outputs

        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - Model Settings" )
        self.Print_Log( "                               - Network Model             : " + str( self.network_model            ) )
        self.Print_Log( "                               - Learning Rate             : " + str( self.learning_rate            ) )
        self.Print_Log( "                               - Dropout                   : " + str( self.dropout                  ) )
        self.Print_Log( "                               - Momentum                  : " + str( self.momentum                 ) )
        self.Print_Log( "                               - Optimizer                 : " + str( self.optimizer                ) )
        self.Print_Log( "                               - # Of Primary Embeddings   : " + str( number_of_unique_terms        ) )
        self.Print_Log( "                               - No. of Inputs             : " + str( self.number_of_inputs         ) )
        self.Print_Log( "                               - No. of Outputs            : " + str( self.number_of_outputs        ) )
        self.Print_Log( "                               - Embedding Dimension Size  : " + str( self.embedding_dimension_size ) )
        self.Print_Log( "                               - No. of NER Outputs        : " + str( number_of_ner_outputs         ) )
        self.Print_Log( "                               - No. of CL Outputs         : " + str( number_of_cl_outputs          ) )
        self.Print_Log( "                               - Trainable Weights         : " + str( self.trainable_weights        ) )

        # Check We're Using The Appropriate Loss Function
        loss_function_a         = { "NER_Output_Layer": "sparse_categorical_crossentropy", "CL_Output_Layer": "binary_crossentropy" }
        loss_function_b         = { "NER_Output_Layer": "sparse_categorical_crossentropy", "CL_Output_Layer": "categorical_crossentropy" }
        cl_activation_functions = [ "sigmoid", "softmax" ]

        if self.loss_function != loss_function_a and self.loss_function != loss_function_b:
            self.Print_Log( "BERTModel::Build_Model() - Warning: Only 'NER Sparse Categorical Crossentropy Loss + CL Categorical/Binary Crossentropy Loss' Functions Are Supported", force_print = True )

            if self.activation_function == "sigmoid":
                self.Print_Log( "BERTModel::Build_Model() - CL Activation Function 'sigmoid' Specified", force_print = True )
                self.Print_Log( "BERTModel::Build_Model() - Setting CL Loss Functions: " + str( loss_function_a ), force_print = True )
                self.loss_function = loss_function_a
            elif self.activation_function == "softmax":
                self.Print_Log( "BERTModel::Build_Model() - CL Activation Function 'softmax' Specified", force_print = True )
                self.Print_Log( "BERTModel::Build_Model() - Setting CL Loss Functions: " + str( loss_function_b ), force_print = True )
                self.loss_function = loss_function_b

        if self.activation_function not in cl_activation_functions:
            self.Print_Log( "BERTModel::Build_Model() - Error: Specified CL Activation Function Not Supported", force_print = True )
            self.Print_Log( "BERTModel::Build_Model() -                                             Supported: " + str( cl_activation_functions ),  force_print = True )
            self.Print_Log( "BERTModel::Build_Model() -                                             Specified: " + str( self.activation_function ), force_print = True )
            self.Print_Log( "BERTModel::Build_Model() - Terminating Program" )
            exit()

        #######################
        #                     #
        #  Build BERT Model   #
        #                     #
        #######################

        # Setup BERT Model Configuration
        bert_model          = self.model_path

        # This Is Technically Not Needed As The BERT Configuration Is Automatically Loaded When Calling 'TFBertModel.from_pretrained()'
        self.bert_config    = BertConfig.from_pretrained( bert_model, num_labels = number_of_ner_outputs )

        # Set Embedding Dimension Size
        embedding_dimension_size = self.bert_config.hidden_size
        if self.embedding_dimension_size != embedding_dimension_size: self.embedding_dimension_size = embedding_dimension_size

        # Setup The BERT Model - Determine If We're Loading From A File Or HuggingFace Model Archive By Model Name
        load_from_file      = True if ".bin" in bert_model or "./" in bert_model else False
        Encoder             = TFBertModel.from_pretrained( bert_model, from_pt = load_from_file, config = self.bert_config )
        self.model_encoder_layer = Encoder.bert

        # # Set Last Layer As SoftMax In BERT Encoder
        # Encoder.layers[-1].activation = tf.keras.activations.softmax

        # Determine If We're Refining BERT Layers In Addition To The Attached Layers Or Just Training The Attached Layers
        #   i.e. Set Encoder Layer Weights/Variables To Trainable Or Freeze Them
        Encoder.trainable   = self.trainable_weights
        # Encoder.layers[-2].activation = True
        # Encoder.layers[-1].activation = True

        # BERT Model Inputs
        token_input_ids     = Input( shape = ( max_sequence_length, ), name = "Token_ID_Input",        dtype = tf.int32 )
        attention_mask      = Input( shape = ( max_sequence_length, ), name = "Attention_Mask_Input",  dtype = tf.int32 )
        token_type_ids      = Input( shape = ( max_sequence_length, ), name = "Token_Type_ID_Input",   dtype = tf.int32 )

        # Pass BERT-Specific Inputs To The Encoder, Extract Embeddings Per Input Sequence Sub-Word
        #   NOTE: Calling "Encoder( input_ids, ... )" Achieves The Same, But Results In A Nested Layer Error When Saving, Then Loading The Model.
        #         Calling "Encoder.bert( inputs_ids, ... )" Resolves The Nested Layer Issue.
        #         Source: https://github.com/keras-team/keras/issues/14345
        encoding_layer      = Encoder.bert( input_ids = token_input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids )[0]   # Same as 'embedding.last_hidden_state' In TFBertModel

        # NER Output Layer
        ner_dropout_layer   = Dropout( name = "NER_Dropout_Layer", rate = self.dropout )( encoding_layer )
        ner_output_layer    = Dense( units = number_of_ner_outputs, activation = "softmax", name = "NER_Output_Layer" )( ner_dropout_layer )

        # Concept Linking Output Layer
        dense_layer         = Dense( units = embedding_dimension_size, activation = 'relu', name = "CL_Dense_Layer_1" )( encoding_layer )
        dropout_layer       = Dropout( name = "CL_Dropout_Layer", rate = self.dropout )( dense_layer )
        cl_output_layer     = Dense( units = number_of_cl_outputs, activation = self.activation_function, name = "CL_Output_Layer" )( dropout_layer )

        self.model = Model( inputs = [token_input_ids, attention_mask, token_type_ids], outputs = [ner_output_layer, cl_output_layer], name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( learning_rate = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt,
                                sample_weight_mode = "temporal", metrics = [ "accuracy" ] )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( learning_rate = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd,
                                sample_weight_mode = "temporal", metrics = [ "accuracy" ] )

        # Print Model Summary
        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =========================================================" )
        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x: self.Print_Log( "NERCLMultiTaskModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =========================================================" )
        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =                                                       =" )
        self.Print_Log( "NERCLMultiTaskModel::Build_Model() - =========================================================" )

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
    print( "     from models import NERCLMultiTaskModel\n" )
    print( "     model = NERCLMultiTaskModel( print_debug_log = True, per_epoch_saving = False, use_csr_format = True )" )
    print( "     model.Fit( \"data/cui_mini\", epochs = 30, batch_size = 4, verbose = 1 )" )
    exit()

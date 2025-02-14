#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    01/15/2021                                                                   #
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
    import tensorflow.keras.backend as K
    from tensorflow.keras        import optimizers
    from tensorflow.keras        import regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Lambda, TimeDistributed
else:
    import keras.backend as K
    from keras        import optimizers
    from keras        import regularizers
    from keras.models import Model
    from keras.layers import Dense, Activation, Input, Concatenate, Dropout, Embedding, Flatten, BatchNormalization, Average, Multiply, Lambda, TimeDistributed

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
        X_batch = self.X[batch_indices,:]
        Y_batch = self.Y[batch_indices,:].todense()

        if self.sample_weights is not None:
            if isinstance( self.sample_weights, COO ) or isinstance( self.sample_weights, csr_matrix ):
                sample_weights = self.sample_weights[batch_indices,:].todense()
            else:
                sample_weights = self.sample_weights[batch_indices,:]
        else:
            sample_weights = None

        return [X_batch, Y_batch], Y_batch, sample_weights


############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class CLMLPModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, margin = 30.0, optimizer = 'adam', activation_function = 'softmax',
                  loss_function = "categorical_crossentropy", number_of_hidden_dimensions = 200, final_layer_type = "dense",
                  embedding_dimension_size = 200, learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.1, batch_size = 32, scale = 0.35,
                  prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0",
                  verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_monitor = "val_loss",
                  early_stopping_patience = 3, use_batch_normalization = False, trainable_weights = False, embedding_a_path = "", embedding_b_path = "",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, class_weights = None, sample_weights = None, use_cosine_annealing = False,
                  cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4 ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, optimizer = optimizer, activation_function = activation_function,
                          loss_function = loss_function, number_of_hidden_dimensions = number_of_hidden_dimensions, prediction_threshold = prediction_threshold,
                          shuffle = shuffle, use_csr_format = use_csr_format, batch_size = batch_size, embedding_dimension_size = embedding_dimension_size,
                          learning_rate = learning_rate, epochs = epochs, per_epoch_saving = per_epoch_saving, use_gpu = use_gpu, momentum = momentum,
                          device_name = device_name, verbose = verbose, debug_log_file_handle = debug_log_file_handle, dropout = dropout,
                          enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, scale = scale,
                          early_stopping_monitor = early_stopping_monitor, margin = margin, final_layer_type = final_layer_type,
                          early_stopping_patience = early_stopping_patience, use_batch_normalization = use_batch_normalization, sample_weights = sample_weights,
                          trainable_weights = trainable_weights, embedding_a_path = embedding_a_path, embedding_b_path = embedding_b_path,
                          feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, class_weights = class_weights,
                          use_cosine_annealing = use_cosine_annealing, cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max )
        self.version       = 0.15
        self.network_model = "concept_linking"


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Converts Randomized Batches Of Model Inputs & Outputs From CSR_Matrix Format
          To Numpy Arrays For Model Training

        Inputs:
            X               : Model Inputs (CSR_Matrix)
            Y               : Model Outputs (CSR_Matrix)
            batch_size      : Batch Size (Integer)
            steps_per_batch : Number Of Iterations Per Epoch (Integer)
            shuffle         : Shuffles Data Prior To Conversion (Boolean)

        Outputs:
            X_batch         : Numpy 2D Matrix Of Model Inputs (Numpy Array)
            Y_input         : Numpy 2D Matrix Of Model Outputs (Numpy Array)
            Y_batch         : Numpy 2D Matrix Of Model Outputs (Numpy Array)

            Modification Of Code From Source: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
    """
    def Batch_Generator( self, X, Y, batch_size, steps_per_batch, shuffle ):
        number_of_instances = X.shape[0]      # Should Be The Same As 'self.trained_instances'
        counter             = 0
        sample_index        = np.arange( number_of_instances )

        if shuffle:
            np.random.shuffle( sample_index )

        while True:
            start_index = batch_size * counter
            end_index   = batch_size * ( counter + 1 )

            # Check - Fixes Batch_Generator Training Errors With The Number Of Instances % Batch Sizes != 0
            end_index   = number_of_instances if end_index > number_of_instances else end_index

            batch_index = sample_index[start_index:end_index]
            X_batch     = X[batch_index,:]
            Y_input     = Y[batch_index,:].todense()
            Y_output    = Y_input
            counter     += 1

            yield [X_batch, Y_input], Y_output

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
        if inputs is None or outputs is None or len( inputs ) == 0 or len( outputs ) == 0:
            self.Print_Log( "CLMLPModel::Fit() - Error: Expected Input/Output Data Length Mismatch", force_print = True )
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

        # Check For Validation Data - Note: We're Not Checking The Output Data, We're Assuming It's Encoded With The Input Data
        if val_inputs is not None and val_outputs is not None:
            if isinstance( val_inputs, list       ) and len( val_inputs )   == 0 or \
               isinstance( val_inputs, COO        ) and val_inputs.shape[0] == 0 or \
               isinstance( val_inputs, csr_matrix ) and val_inputs.shape[0] == 0:
                validation_data = None
            else:
                validation_data = ( [val_inputs, val_outputs], val_outputs )
        else:
            validation_data = None

        if self.use_csr_format:
            self.trained_instances     = inputs.shape[0]
            number_of_input_instances  = inputs.shape[0]
            number_of_output_instances = outputs.shape[0]
        else:
            self.trained_instances     = len( inputs )
            number_of_input_instances  = len( inputs )
            number_of_output_instances = len( outputs )

        self.Print_Log( "CLMLPModel::Fit() - Model Training Settings" )
        self.Print_Log( "                  - Epochs             : " + str( self.epochs             ) )
        self.Print_Log( "                  - Batch Size         : " + str( self.batch_size         ) )
        self.Print_Log( "                  - Verbose            : " + str( self.verbose            ) )
        self.Print_Log( "                  - Shuffle            : " + str( self.shuffle            ) )
        self.Print_Log( "                  - Use CSR Format     : " + str( self.use_csr_format     ) )
        self.Print_Log( "                  - Per Epoch Saving   : " + str( self.per_epoch_saving   ) )
        self.Print_Log( "                  - No. of Train Inputs: " + str( self.trained_instances  ) )

        # Data Generator Placeholders
        data_generator, validation_generator = None, None

        # Compute Number Of Steps Per Batch (Use CSR Format == True)
        steps_per_batch, val_steps_per_batch = 0, None

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
        self.Print_Log( "CLMLPModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            # This Is Depreciated In Tensorflow Versions Greater Than 2.x
            if self.Get_Is_Using_TF1() and self.use_csr_format:
                # Prepare Model Training Data Generator
                data_generator = self.Batch_Generator( inputs, outputs, batch_size = self.batch_size,
                                                       steps_per_batch = steps_per_batch, shuffle = self.shuffle )
                # Prepare Model Validation Data Generator
                if validation_data:
                    validation_generator = self.Batch_Generator( val_inputs, val_outputs, batch_size = self.batch_size,
                                                                 steps_per_batch = val_steps_per_batch, shuffle = False )

                self.model_history = self.model.fit_generator( generator = data_generator, validation_data = validation_generator, epochs = self.epochs,
                                                               steps_per_epoch = steps_per_batch, validation_steps = val_steps_per_batch, verbose = self.verbose,
                                                               callbacks = self.callback_list, class_weight = self.class_weights )
            elif self.use_csr_format:
                data_generator = ModelDataGenerator( X = inputs, Y = outputs, batch_size = self.batch_size, number_of_instances = number_of_input_instances,
                                                     shuffle = self.shuffle, sample_weights = self.Get_Sample_Weights() )

                if validation_data:
                    validation_instances = validation_data[0][0].shape[0]
                    validation_generator = ModelDataGenerator( X = val_inputs, Y = val_outputs, batch_size = self.batch_size, number_of_instances = validation_instances, shuffle = self.shuffle )

                self.model_history = self.model.fit( data_generator, validation_data = validation_generator, shuffle = self.shuffle, batch_size = self.batch_size,
                                                     epochs = self.epochs, verbose = self.verbose, callbacks = self.callback_list, class_weight = self.Get_Class_Weights() )
            else:
                self.model_history = self.model.fit( [inputs, outputs], outputs, validation_data = validation_data, shuffle = self.shuffle, batch_size = self.batch_size,
                                                      epochs = self.epochs, verbose = self.verbose, callbacks = self.callback_list, class_weight = self.Get_Class_Weights(),
                                                      sample_weight = self.Get_Sample_Weights() )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "CLMLPModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "CLMLPModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "CLMLPModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "CLMLPModel::Fit() - Complete" )
        return True

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input_vector   : Vectorized Primary Model Input (Numpy Array)
            secondary_input_vector : Vectorized Secondary Model Input (Numpy Array)
            verbose                : Sets Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)

        Outputs:
            prediction             : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, inputs, verbose = 0 ):
        # Check(s)
        if inputs is None or len( inputs ) == 0:
            self.Print_Log( "CLMLPModel::Predict() - Error: Expected InputData Length Mismatch", force_print = True )
            return False

        if isinstance( inputs, list ): inputs = np.asarray( inputs )

        # Tensorflow 2.x Model Input Conversion
        if self.Get_Is_Using_TF2():
            if isinstance( inputs, COO ) or isinstance( inputs, csr_matrix ): inputs = inputs.todense()

        self.Print_Log( "CLMLPModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            # Give The Network False Output Instance Since They're Not Used For Inference Anyway
            cosface_inputs = np.zeros( ( inputs.shape[0], self.Get_Number_Of_Outputs() ), dtype = np.int32 )
            return self.model.predict( [inputs, cosface_inputs], batch_size = self.batch_size, verbose = verbose )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            inputs  : Encoded Input Matrix (CSR_Matrix, COO, Matrix or Numpy Array)

        Outputs:
            Metrics : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs, outputs, verbose ):
        # Check(s)
        if inputs is None or outputs is None or len( inputs ) == 0 or len( outputs ) == 0:
            self.Print_Log( "CLMLPModel::Evaluate() - Error: Expected Input/Output Data Length Mismatch", force_print = True )
            return -1, -1, -1, -1, -1

        # Tensorflow 2.x Model Input Conversion
        if self.Get_Is_Using_TF2():
            if isinstance( inputs,  COO ) or isinstance( inputs,  csr_matrix ): inputs  = inputs.todense()
            if isinstance( outputs, COO ) or isinstance( outputs, csr_matrix ): outputs = outputs.todense()

        self.Print_Log( "CLMLPModel::Evaluate() - Executing Model Evaluation" )

        with tf.device( self.device_name ):
            # Give The Network False Output Instance Since They're Not Used For Inference Anyway
            loss, accuracy, precision, recall, f1_score = self.model.evaluate( [inputs, outputs], outputs, verbose = verbose )

            self.Print_Log( "CLMLPModel::Evaluate() - Complete" )

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
            number_of_inputs         : (Integer)
            number_of_outputs        : (Integer)
            final_layer_type         : Specifies Final Layer - Dense, CosFace, ArcFace, SphereFace (String)
            embeddings               : Embedding Matrix - Used For Embedding Layer (Numpy 2D Array)
            weight_decay             : Weight Decay Constant For L2 Regularizer (Float)

        Outputs:
            None
    """
    def Build_Model( self, embedding_dimension_size, number_of_inputs, number_of_outputs, number_of_unique_terms = 0,
                     embeddings = [], final_layer_type = None, weight_decay = 0.0001, max_sequence_length = 0 ):
        # Check
        if number_of_inputs != 1: number_of_inputs = 1

        # Update 'BaseModel' Class Variables
        if embedding_dimension_size != self.embedding_dimension_size: self.embedding_dimension_size = embedding_dimension_size
        if number_of_inputs         != self.number_of_inputs:         self.number_of_inputs         = number_of_inputs
        if number_of_outputs        != self.number_of_outputs:        self.number_of_outputs        = number_of_outputs
        if final_layer_type         is not None:                      self.final_layer_type         = final_layer_type

        self.Print_Log( "CLMLPModel::Build_Model() - Model Settings" )
        self.Print_Log( "                          - Network Model             : " + str( self.network_model            ) )
        self.Print_Log( "                          - Final Layer Type          : " + str( self.final_layer_type         ) )
        self.Print_Log( "                          - Learning Rate             : " + str( self.learning_rate            ) )
        self.Print_Log( "                          - Dropout                   : " + str( self.dropout                  ) )
        self.Print_Log( "                          - Momentum                  : " + str( self.momentum                 ) )
        self.Print_Log( "                          - Optimizer                 : " + str( self.optimizer                ) )
        self.Print_Log( "                          - Margin                    : " + str( self.margin                   ) )
        self.Print_Log( "                          - Scale                     : " + str( self.scale                    ) )
        self.Print_Log( "                          - Weight Decay              : " + str( weight_decay                  ) )
        self.Print_Log( "                          - # Of Primary Embeddings   : " + str( number_of_unique_terms        ) )
        self.Print_Log( "                          - No. of Inputs             : " + str( self.number_of_inputs         ) )
        self.Print_Log( "                          - Embedding Dimension Size  : " + str( self.embedding_dimension_size ) )
        self.Print_Log( "                          - No. of Outputs            : " + str( self.number_of_outputs        ) )
        self.Print_Log( "                          - Trainable Weights         : " + str( self.trainable_weights        ) )
        self.Print_Log( "                          - Feature Scaling Value     : " + str( self.feature_scale_value      ) )

        # Check(s)
        final_layer_type_list = ["dense", "arcface", "cosface", "sphereface"]

        if self.final_layer_type not in final_layer_type_list:
            self.Print_Log( "CLMLPModel::Build_Model() - Error: Invalid Final Layer Type", force_print = True )
            self.Print_Log( "                          - Options: " + str( final_layer_type_list ), force_print = True )
            self.Print_Log( "                          - Specified Option: " + str( self.final_layer_type ), force_print = True )
            return False

        #######################
        #                     #
        #  Build Keras Model  #
        #                     #
        #######################

        input_layer         = Input( shape = ( number_of_inputs,  ), name = "Term_ID_Input" )
        cosface_input_layer = Input( shape = ( number_of_outputs, ), name = "CosFace_Input" )

        # Add Embeddings Or Embeddings Initialized As Random Weights
        if len( embeddings ) > 0:
            embedding_layer = Embedding( number_of_unique_terms, embedding_dimension_size, input_length = 1, name = 'Term_Embedding_Layer', weights = [embeddings], trainable = self.trainable_weights )( input_layer )
        else:
            embedding_layer = Embedding( number_of_unique_terms, embedding_dimension_size, input_length = 1, name = 'Term_Embedding_Layer', trainable = self.trainable_weights )( input_layer )

        flatten_layer = Flatten( name = "Embedding_Dimensionality_Reduction" )( embedding_layer )

        # Perform Feature Scaling Prior To Generating An Embedding Representation
        if self.feature_scale_value != 1.0:
            feature_scale_value = self.feature_scale_value  # Fixes Python Recursion Limit Error (Model Tries To Save All 'self' Variable When Used With Lambda Function)
            flatten_layer       = Lambda( lambda x: x * feature_scale_value )( flatten_layer )

        dense_layer       = None
        batch_norm_layer  = None
        dropout_layer     = None
        output_layer      = None

        if self.use_batch_normalization:
            dense_layer       = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_1' )( flatten_layer )
            batch_norm_layer  = BatchNormalization( name = "Batch_Norm_Layer_1" )( dense_layer )
            dropout_layer     = Dropout( name = "Dropout_Layer_2", rate = self.dropout )( batch_norm_layer )
            dense_layer       = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_2',
                                       kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2( weight_decay ) )( dropout_layer )
            dense_layer       = BatchNormalization( name = "Batch_Norm_Layer_2" )( dense_layer )
        else:
            dense_layer       = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_1' )( flatten_layer )
            dropout_layer     = Dropout( name = "Dropout_Layer_2", rate = self.dropout )( dense_layer )
            dense_layer       = Dense( units = embedding_dimension_size, input_dim = embedding_dimension_size, activation = 'relu', name = 'Forward_Feed_Dense_Layer_2',
                                       kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2( weight_decay ) )( dropout_layer )

        # Final Model Output Used For Prediction
        if self.final_layer_type == "arcface":
            output_layer = ArcFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( weight_decay ) )( [dense_layer, cosface_input_layer] )
        elif self.final_layer_type == "sphereface":
            output_layer = SphereFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( weight_decay ) )( [dense_layer, cosface_input_layer] )
        elif self.final_layer_type == "cosface":
            output_layer = CosFace( number_of_outputs, margin = self.margin, scale = self.scale, activation = "multi_label", regularizer = regularizers.l2( weight_decay ) )( [dense_layer, cosface_input_layer] )
        else:
            output_layer = Dense( units = number_of_outputs, input_dim = embedding_dimension_size, activation = self.activation_function, name = "Output_Layer" )( dense_layer )

        self.model = Model( inputs = [input_layer, cosface_input_layer], outputs = output_layer, name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( learning_rate = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt,
                                metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( learning_rate = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd,
                                metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )

        # Print Model Summary
        self.Print_Log( "CLMLPModel::Build_Model() - =========================================================" )
        self.Print_Log( "CLMLPModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "CLMLPModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x: self.Print_Log( "CLMLPModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "CLMLPModel::Build_Model() - =========================================================" )
        self.Print_Log( "CLMLPModel::Build_Model() - =                                                       =" )
        self.Print_Log( "CLMLPModel::Build_Model() - =========================================================" )

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
    print( "     from models import CLMLPModel\n" )
    print( "     model = CLMLPModel( print_debug_log = True, per_epoch_saving = False, use_csr_format = True )" )
    print( "     model.Fit( \"data/cui_mini\", epochs = 30, batch_size = 4, verbose = 1 )" )
    exit()

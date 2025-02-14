#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/10/2022                                                                   #
#    Revised: 09/13/2022                                                                   #
#                                                                                          #
#    Keras Custom Layers                                                                   #
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
import os, re

# Suppress Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes Tensorflow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow as tf

#tf.logging.set_verbosity( tf.logging.ERROR )                       # Tensorflow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # Tensorflow v1.x

if re.search( r"^2.\d+", tf.__version__ ):
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers  import Layer
    from tensorflow.keras         import regularizers
    from tensorflow.keras.metrics import categorical_accuracy
else:
    import keras.backend as K
    from keras.layers  import Layer
    from keras         import regularizers
    from keras.metrics import categorical_accuracy


############################################################################################
#                                                                                          #
#    Custom BERT Embedding Extraction Layer                                                #
#                                                                                          #
############################################################################################

"""
    Receives BERT Sub-Word Embeddings For A Given Input Sequence And Uses A Mask To
       Zero-Out Un-Desired Sub-Word Embeddings While Preserving Desired Sub-Word Embeddings.

    Example: Desired Embedding Representation - 'nalaxone'
             Input Sequence: 'nalaxone reverses the antihypertensive effects of clonidine.'
             Tokenized Sub-Words: na, ##lo, ##xon, ##e, reverse, ##s, the, anti, ##hy, ##pert, ##ens, ##ive, effect, of, c, ##lon, ##ine, .

    All Sub-Word Embeddings Including 'na', '##lo', '##xon' & '##e' Are Averaged And Returned

    Input Sub-Word Embedding Shape: ( batch_size, sequence_length, hidden_size )
    Averaged Embedding Shape      : ( batch_size, hidden_size )

    This Layer Is Also Able To Output The First Or Last Embeddings Within The Masked Sequence Depending On User Settings.
      Output Embdding Shape Remains The Same Shape With All Three Options.

    NOTE: All Three Methods Require More Testing. Model Reaches Saturation With Repeat Epochs.
          (Loss Fluctuates As Well As Reported Model Generalization Metrics).

    Inputs:
        output_embedding_type : 'average', 'first' or 'last' (String)
        hidden_size           : Embedding Size (Int)

    Outputs:
        embedding             : Embedding Of Shape ( batch_size, hidden_size )
"""
class Embedding_Extraction_Layer( Layer ):
    def __init__( self, output_embedding_type = "average", hidden_size = 768, **kwargs ):
        self.version               = 0.01
        self.mask_shape            = None
        self.embedding_shape       = None
        self.hidden_size           = hidden_size
        self.output_embedding_type = output_embedding_type
        super( Embedding_Extraction_Layer, self ).__init__( **kwargs )

    def build( self, input_shape ):
        self.mask_shape      = input_shape[0]   # Saved For Future Use
        self.embedding_shape = input_shape[1]   # Saved For Future Use
        super( Embedding_Extraction_Layer, self ).build( input_shape )

    def call( self, inputs ):
        # Separate The Entry Term Mask From The Sub-Word Embeddings
        sub_word_embeddings, entry_term_mask = inputs

        # Check If The Entry Term Mask Array Is Two Dimension
        #   If So, Reshape It To Three Dimensions To Match Our Sub Word Embedding Array/Vectors
        #   Note: This Should Not Be Called. Entry Term Mask Should Be Reshaped Within The Model's Build_Model Function.
        if K.ndim( entry_term_mask ) == 1: entry_term_mask = tf.reshape( entry_term_mask,
                                                                         shape = ( entry_term_mask.shape[0], entry_term_mask.shape[-1], 1 ) )

        # Cast The Entry Term Mask From Int32 To Float32
        #   Next, Apply The Entry Term Mask To The Sub-Word Embeddings
        #   This Should Only Leave The Entry-Term Sub-Words As Non-Zero Entries
        sub_word_embeddings = tf.cast( entry_term_mask, dtype = tf.float32 ) * sub_word_embeddings

        output_embedding = None

        ######################################################################################################
        # Return First Non-Zeroed Sub-Word Of The Non-Zeroed Sub-Word Embeddings                             #
        #   Shape Reduces From: ( batch_size, sequence_length, hidden_size ) to ( batch_size, hidden_size )  #
        #                   ie: ( None, 512, 768 ) to ( None, 768 )                                          #
        ######################################################################################################
        if self.output_embedding_type == "first":
            # First Sum All The Masked Sub-Word Embeddings
            #   Shape ( batch_size, sequence_length )
            summed_embeddings = tf.cast( tf.reduce_sum( sub_word_embeddings, axis = -1 ), dtype = tf.float32 )

            # Determine Indices Of Non-Zero Embeddings
            #   Shape ( batch_size, sequence_length ) i.e. ( batch_size, num_of_non_zero_indices )
            non_zero_indices  = tf.cast( tf.not_equal( summed_embeddings, 0 ), dtype = tf.int32 )

            #   Mask Sequence Indices With A Matrix Where All Elements == non_zero_indices.shape[-1].             (y Variable)
            #   Next, For Each Non-Zero Element In The Equality Matrix, Apply Values In Ascending Order By Column (x Variable)
            #   Return Back Combined Matrix
            #      i.e. non_zero_indices = [[ 1, 0, 0, 1],    y = [[ 5, 5, 5, 5],   output_matrix = [[ 0, 5, 5, 3],
            #                               [ 0, 0, 1, 0]]         [ 5, 5, 5, 5]]                    [ 5, 5, 2, 5]]
            #   The Elements: '0', '2' and '3' In The Output Matrix Represent Instance Indices Which Are Non-Zero
            #
            #   Shape ( batch_size, sequence_length )
            non_zero_indices  = tf.where( tf.equal( non_zero_indices, 1 ),
                                          x = tf.range( tf.shape( non_zero_indices )[-1] ),
                                          y = tf.shape( non_zero_indices )[-1] + 1 )

            # Extract The First Non-Zero Element Index From Each Instance
            #   Shape ( batch_size, 1 ) i.e. ( batch_size, first_index_where_element_count_is_not_zero )
            #   i.e. first_indices = [[0], [2]]
            first_indices     = tf.reduce_min( non_zero_indices, axis = -1, keepdims = True )

            #   Count The Number Of Instances Within Our Batch
            #   Shape ( batch_size, 1 )
            instance_counts   = tf.reshape( tf.range( tf.shape( first_indices )[0] ), shape = tf.shape( first_indices ) )

            #   Stack The Instance Counts To Our 'first_indices' Elements
            #     i.e. first_indices   = [[0], [2]]
            #          instance_counts = [[0], [1]]
            #          first_indices   = [[0, 0], [1, 2]]
            first_indices     = tf.stack( [instance_counts, first_indices], axis = -1 )
            first_indices     = tf.reshape( first_indices, shape = ( -1, first_indices.shape[-1] ) )

            # Store First Embedding For Each Instance In The Batch Using Its Index
            #   Shape ( batch_size, hidden_size )
            output_embedding  = tf.gather_nd( sub_word_embeddings, first_indices )

        ######################################################################################################
        # Return Last Non-Zeroed Sub-Word Of The Non-Zeroed Sub-Word Embeddings                              #
        #   Shape Reduces From: ( batch_size, sequence_length, hidden_size ) to ( batch_size, hidden_size )  #
        #                   ie: ( None, 512, 768 ) to ( None, 768 )                                          #
        ######################################################################################################
        elif self.output_embedding_type == "last":
            # First Sum All The Masked Sub-Word Embeddings
            #   Shape ( batch_size, sequence_length )
            summed_embeddings = tf.cast( tf.reduce_sum( sub_word_embeddings, axis = -1 ), dtype = tf.float32 )

            # Determine Indices Of Non-Zero Embeddings
            #   Shape ( batch_size, sequence_length ) i.e. ( batch_size, num_of_non_zero_indices )
            non_zero_indices  = tf.cast( tf.not_equal( summed_embeddings, 0 ), dtype = tf.int32 )

            #   Mask Sequence Indices With A Matrix Where All Elements == -1.   (y Variable)
            #   Next, For Each Non-Zero Element In The Equality Matrix, Apply Values In Ascending Order By Column (x Variable)
            #   Return Back Combined Matrix
            #      i.e. non_zero_indices = [[ 1, 0, 0, 1],    y = [[ -1, -1, -1, -1],   output_matrix = [[  0, -1, -1,  3],
            #                               [ 0, 0, 1, 0]]         [ -1, -1, -1, -1]]                    [ -1, -1,  2, -1]]
            #   The Elements: '0', '1' and '3' In The Output Matrix Represent Instance Indices Which Are Non-Zero
            #
            #   Shape ( batch_size, sequence_length )
            non_zero_indices  = tf.where( tf.equal( non_zero_indices, 1 ),
                                          x = tf.range( tf.shape( non_zero_indices )[-1] ),
                                          y = -1 )

            # Extract The First Non-Zero Element Index From Each Instance
            #   Shape ( batch_size, 1 ) i.e. ( batch_size, first_index_where_element_count_is_not_zero )
            #   i.e. last_indices = [[3], [2]]
            last_indices      = tf.reduce_max( non_zero_indices, axis = -1, keepdims = True )

            #   Count The Number Of Instances Within Our Batch
            #   Shape ( batch_size, 1 )
            instance_counts   = tf.reshape( tf.range( tf.shape( last_indices )[0] ), shape = tf.shape( last_indices ) )

            #   Stack The Instance Counts To Our 'last_indices' Elements
            #     i.e. last_indices    = [[3], [2]]
            #          instance_counts = [[0], [1]]
            #          last_indices    = [[0, 3], [1, 2]]
            last_indices      = tf.stack( [instance_counts, last_indices], axis = -1 )
            last_indices      = tf.reshape( last_indices, shape = ( -1, last_indices.shape[-1] ) )

            # Store Last Embedding For Each Instance In The Batch Using Its Index
            #   Shape ( batch_size, hidden_size )
            output_embedding  = tf.gather_nd( sub_word_embeddings, last_indices )

        ######################################################################################################
        # Return The Mean Of The Non-Zeroed Sub-Word Embeddings                                              #
        #   Shape Reduces From: ( batch_size, sequence_length, hidden_size ) to ( batch_size, hidden_size )  #
        #                   ie: ( None, 512, 768 ) to ( None, 768 )                                          #
        ######################################################################################################
        elif self.output_embedding_type == "average":
            # Count The Number Of Non-Zero Embeddings
            #   Shape ( batch_size, 1, non_zero_element_count_value )
            #   NOTE: Cleaner Solution: Use 'tf.math.count_nonzero()' Function
            num_of_non_zero_embeddings = tf.reduce_sum( tf.cast( tf.not_equal( entry_term_mask, 0 ), dtype = tf.float32 ), axis = 1, keepdims = True )

            # Sum All The Sub-Word Embeddings / This Really Sums All Entry-Term Sub-Words Since They're The Only Non-Zero Embeddings Left
            #   Shape ( batch_size, 1, hidden_size )
            summed_sub_word_embeddings = tf.reduce_sum( sub_word_embeddings, axis = 1, keepdims = True )

            # Compute The Mean Of All Entry-Term Sub-Word Embeddings
            #   Shape ( batch_size, 1, hidden_size )
            output_embedding = summed_sub_word_embeddings / num_of_non_zero_embeddings

            # Reshape Averaged Output
            #   Shape ( batch_size, hidden_size )
            output_embedding = tf.reshape( output_embedding, shape = ( -1, output_embedding.shape[-1] ) )

        # -------------------------------------------------------- #
        #                 Debug Function Calls                     #
        # -------------------------------------------------------- #

        # Used For Debugging Purposes - Detected 'NaN' Element In Output Embedding And Prints Necessary Variables To Debug
        #   Leave Commented If Not Needed. Functions In The 'if' Statement Append New Elements To The Compute Graph Each Call,
        #   Leading To Memory Consumption Issues As Epochs Increase.
        #
        # if tf.reduce_any( tf.math.is_nan( output_embedding ) ):
        #     tf.print( "Sub Word Embeddings: ", sub_word_embeddings, summarize = -1 )
        #     tf.print( "Entry Term Mask: ", tf.reshape( entry_term_mask, shape = ( -1, entry_term_mask.shape[1] ) ), summarize = -1 )
        #     tf.print( "Output Embedding: ", output_embedding, summarize = -1 )

        # Return Embedding
        return output_embedding

    # Not Used
    # def compute_mask( self, input_shape, input_mask = None ):
    #     return None

    def compute_output_shape( self, input_shape ):
        assert len( input_shape ) == 3
        return ( input_shape[0], self.hidden_size )

    def get_config( self ):
        config = super().get_config()
        config.update({
            "hidden_size"          : self.hidden_size,
            "output_embedding_type": self.output_embedding_type,
        })
        return config

############################################################################################
#                                                                                          #
#    ArcFace Class                                                                         #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class ArcFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 0.50, regularizer = None, activation = "softmax", **kwargs ):
        super( ArcFace, self ).__init__( **kwargs )
        self.version     = 0.01
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( ArcFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'ArcFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape( x )[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos( K.clip( logits, -1.0 + K.epsilon(), 1.0 - K.epsilon() ) )
        target_logits = tf.cos( theta + self.margin )
        # sin = tf.sqrt( 1 - logits ** 2 )
        # cos_m = tf.cos( logits )
        # sin_m = tf.sin( logits )
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )

    def get_config( self ):
        config = super().get_config()
        config.update({
            "n_classes"  : self.n_classes,
            "scale"      : self.scale,
            "margin"     : self.margin,
            "regularizer": self.regularizer,
            "activation" : self.activation,
        })
        return config


############################################################################################
#                                                                                          #
#    CosFace Class                                                                         #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class CosFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 0.35, regularizer = None, activation = "softmax", **kwargs ):
        super( CosFace, self ).__init__( **kwargs )
        self.version     = 0.01
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( CosFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'CosFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape( x )[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.margin
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )

    def get_config( self ):
        config = super().get_config()
        config.update({
            "n_classes"  : self.n_classes,
            "scale"      : self.scale,
            "margin"     : self.margin,
            "regularizer": self.regularizer,
            "activation" : self.activation,
        })
        return config


############################################################################################
#                                                                                          #
#    SphereFace Class                                                                      #
#                                                                                          #
#        Modified From Source:  https://github.com/4uiiurz1/keras-arcface                  #
#                                                                                          #
############################################################################################

class SphereFace( Layer ):
    def __init__( self, n_classes = 10, scale = 30.0, margin = 1.35, regularizer = None, activation = "softmax", **kwargs ):
        super( SphereFace, self ).__init__( **kwargs )
        self.version     = 0.01
        self.n_classes   = n_classes
        self.scale       = scale
        self.margin      = margin
        self.activation  = activation
        self.regularizer = regularizers.get( regularizer )

    def build( self, input_shape ):
        super( SphereFace, self ).build( input_shape[0] )
        self.W = self.add_weight( name        = 'SphereFace_Weights',
                                  shape       = ( input_shape[0][-1], self.n_classes ),
                                  initializer = 'glorot_uniform',
                                  trainable   = True,
                                  regularizer = self.regularizer )

    def call( self, inputs ):
        x, y = inputs

        # Used For Debugging Purposes - Prints Ground Truth Labels Per Class
        # y = tf.Print( y, [tf.round(y)], summarize = -1 )

        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize( x, axis = 1 )
        # normalize weights
        W = tf.nn.l2_normalize( self.W, axis = 0 )
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos( K.clip( logits, -1.0 + K.epsilon(), 1.0 - K.epsilon() ) )
        target_logits = tf.cos( self.margin * theta )
        #
        logits = logits * ( 1 - y ) + target_logits * y
        # feature re-scale
        logits *= self.scale

        predictions = None

        # Pass Logits Through Softmax Function
        #   ie. Generate Normalized Distribution Over All Logits Between 0.0 and 1.0
        if self.activation == "softmax":
            predictions = tf.nn.softmax( logits )
        # Pass Each Logit Element Through The Sigmoid Function To Generate A Multi-Class/Multi-Label Distribution Per Element
        else:
            predictions = 1/( 1 + tf.math.exp( ( -logits ) ) )

        # Used For Debugging Purposes - Prints Predicted Labels Per Class
        # predictions = tf.Print( predictions, [tf.round(predictions)], summarize = -1 )

        return predictions

    def compute_output_shape( self, input_shape ):
        return ( None, self.n_classes )

    def get_config( self ):
        config = super().get_config()
        config.update({
            "n_classes"  : self.n_classes,
            "scale"      : self.scale,
            "margin"     : self.margin,
            "regularizer": self.regularizer,
            "activation" : self.activation,
        })
        return config


############################################################################################
#                                                                                          #
#    ArcFace Alt Class                                                                     #
#                                                                                          #
#        Not My Source. Also Not Verified As Working.                                      #
#                                                                                          #
############################################################################################

class ArcFaceAlt( Layer ):
    '''Custom Keras layer implementing ArcFace including:
    1. Generation of embeddings
    2. Loss function
    3. Accuracy function
    '''

    def __init__(self, output_dim, class_num, margin=0.5, scale=64., **kwargs):
        self.version    = 0.01
        self.output_dim = output_dim
        self.class_num = class_num
        self.margin = margin
        self.s = scale

        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.mm = self.sin_m * margin
        self.threshold = tf.math.cos(tf.constant(math.pi) - margin)
        super(ArcFaceAlt, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.class_num),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(ArcFaceAlt, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        embeddings = tf.nn.l2_normalize(x, axis=1, name='normed_embeddings')
        weights = tf.nn.l2_normalize(self.kernel, axis=0, name='normed_weights')
        cos_t = tf.matmul(embeddings, weights, name='cos_t')
        return cos_t

    def get_logits(self, labels, y_pred):
        cos_t = y_pred
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')
        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = self.s*(cos_t - self.mm)
        cos_mt_temp = tf.compat.v1.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(labels, depth=self.class_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')
        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
        return output

    def loss(self, y_true, y_pred):
        labels = K.argmax(y_true, axis=-1)
        logits = self.get_logits(labels, y_pred)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return loss

    def accuracy(self, y_true, y_pred):
        labels = K.argmax(y_true, axis=-1)
        logits = self.get_logits(labels, y_pred)
        accuracy = categorical_accuracy(y_true=labels, y_pred=logits)
        return accuracy

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config( self ):
        config = super().get_config()
        config.update({
            "output_dim" : self.output_dim,
            "class_dim"  : self.class_dim,
            "margin"     : self.margin,
            "scale"      : self.scale,
        })
        return config


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Imported From From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from KerasLayers import Embedding_Extraction_Layer\n" )
    print( "     # Build Model And Include 'Embedding_Extraction_Layer()" )
    exit()

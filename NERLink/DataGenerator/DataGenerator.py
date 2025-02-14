#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Named Entity Recognition + Concept Linking Package                                    #
#    --------------------------------------------------                                    #
#                                                                                          #
#    Date:    05/24/2022                                                                   #
#    Revised: 05/24/2022                                                                   #
#                                                                                          #
#    Base Data Generator Class Used For NER And Concept Linking Models.                    #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator( Sequence ):
    'Initialization'
    def __init__( self, X, Y, batch_size = 1, dimensions = ( None, None ),
                  number_of_instances = None, number_of_classes = None,
                  shuffle = True, sample_weights = None ):
        self.X                   = X
        self.Y                   = Y
        self.batch_size          = batch_size
        self.dimensions          = dimensions
        self.number_of_instances = number_of_instances
        self.number_of_classes   = number_of_classes
        self.shuffle             = shuffle
        self.sample_weights      = sample_weights
        self.indices             = None

        # Initialize Instance Indices
        self.on_epoch_end()

    'Number Of Batches Per Epoch'
    def __len__( self ):
        return int( np.ceil( self.number_of_instances / self.batch_size ) )

    'Generate One Batch Of Data'
    def __getitem__( self, index ):
        # Generate Batch Given Start And End Indices
        start_index   = self.batch_size * index
        end_index     = self.batch_size * ( index + 1 )
        batch_indices = self.indices[start_index:end_index]

        # Extract Indices Given Inputs
        X_batch = self.X[batch_indices,:]
        Y_batch = self.Y[batch_indices,:]

        return X_batch, Y_batch, self.sample_weights

    'Shuffles Data After Every Epoch'
    def on_epoch_end( self ):
        self.indices = np.arange( self.number_of_instances )

        if self.shuffle:
            np.random.shuffle( self.indices )
import numpy as np

import os
os.environ['KMP_WARNINGS'] = 'FALSE'
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import tensorflow.keras.losses as losses


def network_actor():
    model = Sequential()
    # The network is used to approximate a POLICY. As a result, we suggest the following structure:
    # 
    # Since the observation space consists of 33 variables corresponding to position, rotation, 
    # velocity, and angular velocities of the arm, the input layer shall consist of 33 neurons
    #
    # The output layer consists of two neurons for four floating point numbers
    # between -1 and 1 which represent the torque applied to the two joints of
    # the robot arm

    model.add( Dense(33, input_shape=(33,) ) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(4) )
    model.add( Activation('tanh') )

    model.compile(loss=losses.mean_squared_error, optimizer='sgd')
    return model


def network_critic():
    model = Sequential()
    # The network is used to approximate a POLICY. As a result, we suggest the following structure:
    # 
    # Since the observation space consists of 33 variables corresponding to position, rotation, 
    # velocity, and angular velocities of the arm, the input layer shall consist of 33 neurons
    #
    # The output layer consists of two neurons for four floating point numbers
    # between -1 and 1 which represent the torque applied to the two joints of
    # the robot arm

    model.add( Dense(33, input_shape=(33,) ) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(64) )
    model.add( Activation('relu') )
    model.add( Dropout(0.2) )
    model.add( Dense(1) )
    model.add( Activation('tanh') )

    model.compile(loss=losses.mean_squared_error, optimizer='sgd')
    return model

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
from keras import layers

# The network is used to approximate a POLICY. As a result, we suggest the following structure:
# 
# Since the observation space consists of 33 variables corresponding to position, rotation, 
# velocity, and angular velocities of the arm, the input layer shall consist of 33 neurons
#
# The output layer consists of two neurons for four floating point numbers
# between -1 and 1 which represent the torque applied to the two joints of
# the robot arm

state_size = 24
action_size = 2
fc1_units = 256
fc2_units = 128


def network_actor():
    state_input = layers.Input( shape=(state_size),) )
    fc1 = layers.Dense(fc1_units, activation='relu' )( state_input )
    fc2 = layers.Dense(fc2_units, activation='relu' )( fc1 )
    action_output = layers.Dense(2, activation='tanh')( fc2 )

    model = Keras.Model(inputs=state_input, outputs=action_output)
    return model


def network_critic():
    state_input = layers.Input( shape=(state_size,) )
    action_input = layers.Input( shape=(action_size,) )
    fc1 = layers.Dense(fc1_units, activation='relu' )( state_input )
    fc2 = layers.Dense(fc2_units, activation='relu' )( [fc1, action_input] )
    action_output = layers.Dense(1, activation='tanh')( fc2 )

    model = Keras.Model(inputs=[state_input, action_input], outputs=value_output)
    return model

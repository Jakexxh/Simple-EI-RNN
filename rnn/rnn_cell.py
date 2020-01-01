import numpy as np
import tensorflow as tf
from tensorflow import keras
from util.util_funs import rectify
import tensorflow.keras.backend as K

from main import SGD_p

UNITS_SIZE = 100
EI_RATIO = 0.8


class EIRNNCell(keras.layers.Layer):

    def __init__(self, **kwargs):
        self.units = UNITS_SIZE
        self.ei_ratio = EI_RATIO

        self.alpha = SGD_p['train_t_step'] / SGD_p['tau']

        self.W_in = None
        self.W_rec = None
        self.W_out = None

        super(EIRNNCell, self).__init__(**kwargs)

        self.is_initialized = False
        self.is_built = False

    def build(self, input_shape, M_rec=None, W_fixed=None):
        self.W_in = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=tf.random_uniform_initializer(minval=0),
                                      name='W_in')

        self.W_out = self.add_weight(shape=(self.units, 2),
                                    initializer=tf.random_uniform_initializer(minval=0),
                                    name='W_out')

        self.W_rec = self.add_weight(shape=(self.units, self.units),
                                     initializer=tf.random_uniform_initializer(minval=0),
                                     name='W_rec')

        self.is_initialized = True
        self.is_built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

    def run_timestep(self, rnn_in, state):
        new_state = ((1 - self.alpha) * state) \
                    + self.alpha * (
                            tf.matmul(
                                self.transfer_function(state),
                                self.get_effective_W_rec(),
                                transpose_b=True, name="1")
                            + tf.matmul(
                        rnn_in,
                        self.get_effective_W_in(),
                        transpose_b=True, name="2")
                            + self.b_rec) \
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) \
                    * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)

        return new_state


# class MinimalRNNCell(keras.layers.Layer):
#
#     def __init__(self, units, **kwargs):
#         self.units = units
#         self.state_size = units
#         super(MinimalRNNCell, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
#                                       initializer='uniform',
#                                       name='kernel')
#         self.recurrent_kernel = self.add_weight(
#             shape=(self.units, self.units),
#             initializer='uniform',
#             name='recurrent_kernel')
#         self.built = True
#
#     def call(self, inputs, states):
#         prev_output = states[0]
#         h = K.dot(inputs, self.kernel)
#         output = h + K.dot(prev_output, self.recurrent_kernel)
#         return output, [output]
#
# # Let's use this cell in a RNN layer:
#
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = keras.layers.RNN(cell)
# y = layer(x)


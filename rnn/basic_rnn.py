
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


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


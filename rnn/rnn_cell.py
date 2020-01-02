import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K

from main import SGD_p

UNITS_SIZE = 100
EI_RATIO = 0.8


class EIRNNCell(keras.layers.Layer):

    def __init__(self, **kwargs):
        self.units = UNITS_SIZE
        self.ei_ratio = EI_RATIO
        self.rho = SGD_p['ini_spe_r']

        self.alpha = SGD_p['train_t_step'] / SGD_p['tau']

        self.W_in = None
        self.W_rec = None
        self.W_out = None

        self._W_rec_plastic = None
        self._M_rec = None
        self._W_fixed = None
        self._M_rec_m = None
        self._W_fixed_m = None

        self.rec_scale = None

        self.Dale_rec = None
        self.Dale_out = None

        self.is_initialized = False
        self.is_built = False

        super(EIRNNCell, self).__init__(**kwargs)

    def build(self, input_shape, M_rec=None, W_fixed=None):

        self.W_in = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=tf.random_uniform_initializer(minval=0),
                                      name='W_in')

        self.W_out = self.add_weight(shape=(self.units, 2),
                                    initializer=tf.random_uniform_initializer(minval=0),
                                    name='W_out')

        self._W_rec_plastic_m = None
        self._W_rec_plastic = self.add_weight(shape=(self.units, self.units),
                                     initializer=tf.constant_initializer(self._W_rec_plastic_m),
                                     name='W_rec_plastic')

        if M_rec is None:
            self._M_rec_m =  np.ones((self.units, self.units)) - np.diag(np.ones(self.units))
            self._M_rec = self.add_weight(shape=(self.units, self.units),
                                           initializer=tf.constant_initializer(self._M_rec_m),
                                           name='M_rec',
                                           trainable=False)

        if W_fixed is None:
            self._W_fixed_m = np.zeros(self.units, self.units)
            self._W_fixed = self.add_weight(shape=(self.units, self.units),
                                            initializer=tf.constant_initializer(self._W_fixed_m),
                                            name='W_fixed',
                                            trainable=False)

        self.rec_scale = self.rho * funs.spectral_radius(np.multiply(self._M_rec_m, self._W_rec_plastic_m)
                                                         + self._W_fixed_m)

        self.W_rec = self.rec_scale * (tf.multiply(self._M_rec, self._W_rec_plastic) + self._W_fixed)

        # Dale
        dale_vec = np.ones(self.units)
        dale_vec[int(self.ei_ratio*self.units):] = -1

        self.Dale_rec= self.add_weight(shape=(self.units, self.units),
                                             initializer=tf.constant_initializer(np.diag(dale_vec)),
                                             name='Dale_rec',
                                             trainable=False)

        self.Dale_out = self.add_weight(shape=(self.units, self.units),
                                        initializer=tf.constant_initializer(np.diag(dale_vec)),
                                        name='Dale_out',
                                        trainable=False)

        self.is_initialized = True
        self.is_built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

    # def run_timestep(self, rnn_in, state):
    #     new_state = ((1 - self.alpha) * state) \
    #                 + self.alpha * (
    #                         tf.matmul(
    #                             self.transfer_function(state),
    #                             self.get_effective_W_rec(),
    #                             transpose_b=True, name="1")
    #                         + tf.matmul(
    #                     rnn_in,
    #                     self.get_effective_W_in(),
    #                     transpose_b=True, name="2")
    #                         + self.b_rec) \
    #                 + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise) \
    #                 * tf.random_normal(tf.shape(state), mean=0.0, stddev=1.0)
    #
    #     return new_state


    def gamma__distribution(self, k, size, theta=1.0):
        init = np.random.gamma(k, theta, size=size)
        rho0 = funs.spectral_radius(init)
        init = (self.rho/rho0)*init
        return tf.constant_initializer(init)

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


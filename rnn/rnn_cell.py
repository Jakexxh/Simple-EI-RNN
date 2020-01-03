import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K

from main import SGD_p


class EIRNNCell(keras.layers.Layer):

    def __init__(self, units_size=100, ei_ratio=0.8, mode='train', **kwargs):
        self.units = units_size
        self.state_size = self.units
        self.ei_ratio = ei_ratio
        self.rho = SGD_p['ini_spe_r']

        if mode == 'train':
            self.alpha = SGD_p['train_t_step'] / SGD_p['tau']
        elif mode == 'test':
            self.alpha = SGD_p['test_t_step'] / SGD_p['tau']
        else:
            raise Exception('Wrong mode')

        self.init_state = None

        self.W_in = None
        self.W_rec = None
        self.W_out = None

        self._W_rec_plastic = None
        self._M_rec = None
        self._W_fixed = None
        self._M_rec_m = None
        self._W_rec_plastic_m = None
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

        self._W_rec_plastic_m = self.glorot_uniform()
        self._W_rec_plastic = self.add_weight(shape=(self.units, self.units),
                                     initializer=tf.constant_initializer(self._W_rec_plastic_m),
                                     name='W_rec_plastic')

        if M_rec is None:
            self._M_rec_m = np.ones((self.units, self.units)) - np.diag(np.ones(self.units))
            self._M_rec = self.add_weight(shape=(self.units, self.units),
                                           initializer=tf.constant_initializer(self._M_rec_m),
                                           name='M_rec',
                                           trainable=False)

        if W_fixed is None:
            self._W_fixed_m = np.zeros((self.units, self.units))
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

        self.Dale_rec = self.add_weight(shape=(self.units, self.units),
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
        x_prev = states[0]
        x = ((1 - self.alpha) * x_prev) + \
            self.alpha * (
                K.dot(K.relu(x_prev), K.dot(self.Dale_rec, self.W_rec)) +
                K.dot(inputs, self.W_in) +
                K.sqrt(K.constant(2.0 * self.alpha * SGD_p['rr_noise_std']**2)) *
                    K.random_normal(K.shape(x_prev)))

        r = K.relu(x)
        z = K.dot(r, K.dot(self.Dale_out, self.W_out))
        return z, [x]


    def glorot_uniform(self, scale=1.0):
        limits = np.sqrt(6 / (self.units + self.units))
        uniform = np.random.uniform(-limits,limits,(self.units,self.units)) * scale
        return np.abs(uniform)

    # def gamma__distribution(self, k, size, theta=1.0):
    #     init = np.random.gamma(k, theta, size=size)
    #     rho0 = funs.spectral_radius(init)
    #     init = (self.rho/rho0)*init
    #     return tf.constant_initializer(init)


# Test
#
# cell = EIRNNCell(100, 0.8)
# x = keras.Input((None, 2))
# layer = keras.layers.RNN(cell)
# y = layer(x)
# pass
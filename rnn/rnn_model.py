import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K
from loss import MaskMeanSquaredError
from rnn import rnn_cell, loss
from main import SGD_p

UNITS_SIZE = 100
EI_RATIO = 0.8
X_0 = 0.1

class SimpleEIRNN:

    def __init__(self):
        self.batch_size = SGD_p['minibatch_size']
        self.lr = SGD_p['lr']
        self.grad_clip = SGD_p['max_grad_norm']

        self.init_state = None
        self.rnn_cell = None
        self.ei_rnn = None

        pass

    def build(self):
        self.init_state = tf.Variable(tf.ones([SGD_p['minibatch_size'], UNITS_SIZE]) * X_0, trainable=True) # TODO: set p to trainble
        self.rnn_cell = rnn_cell.EIRNNCell(UNITS_SIZE, EI_RATIO)
        self.ei_rnn = keras.layers.RNN(self.rnn_cell, return_sequences=False, return_state=False)


    def train(self):
        pass

    def test(self):
        pass


# TODO:
#  Add regular for loss
#  Add clip
#  Add mask for loss
#  Add terminate



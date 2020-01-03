import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K
from rnn import rnn_cell, loss
from main import SGD_p


class SimpleEIRNN:

    def __init__(self):
        self.batch_size = SGD_p['minibatch_size']
        pass

    def build(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


# TODO:
#  Add regular for loss
#  Add mask for loss
#  Add terminate



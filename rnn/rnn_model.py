import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K

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

        self.optimizer = None

        self.loss_fun = loss.MaskMeanSquaredError
        pass

    def build(self):
        self.init_state = tf.Variable(tf.ones([SGD_p['minibatch_size'], UNITS_SIZE]) * X_0, trainable=True) # TODO: set p to trainble
        self.rnn_cell = rnn_cell.EIRNNCell(UNITS_SIZE, EI_RATIO)
        self.ei_rnn = keras.layers.RNN(self.rnn_cell, return_sequences=False, return_state=False)

        self.optimizer = keras.optimizers.SGD(learning_rate=SGD_p['lr'])

    def train(self):
        # for every epoch
        #   for every batch
        #       1. get mask, inputs, outputs, masks
        #       2. run and get loss
        #       3.
        pass

    def test(self):
        pass


# # Iterate over the batches of a dataset.
# for x_batch_train, y_batch_train in train_dataset:
#   with tf.GradientTape() as tape:
#     logits = layer(x_batch_train)  # Logits for this minibatch
#     # Loss value for this minibatch
#     loss_value = loss_fn(y_batch_train, logits)
#     # Add extra losses created during this forward pass:
#     loss_value += sum(model.losses)
#
#   grads = tape.gradient(loss_value, model.trainable_weights)
#   optimizer.apply_gradients(zip(grads, model.trainable_weights))

# TODO:
#  Add clip -> gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#  Add mask for loss
#  Add terminate


"""
Test


"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import util.util_funs as funs
import tensorflow.keras.backend as K
import datetime

from util import plot
from data.data_generator import DataGenerator
from rnn import rnn_cell, loss
from main import SGD_p

UNITS_SIZE = 100
EI_RATIO = 0.8
X_0 = 0.1


class SimpleEIRNN:

    def __init__(self, args):

        self.task_version = args['task_version']
        self.init_state_trainable = args['init_state_trainable']
        self.lr = SGD_p['lr']
        self.grad_clip = SGD_p['max_grad_norm']

        self.init_state = None
        self.rnn_cell = None
        self.ei_rnn = None

        self.optimizer = None
        self.loss_fun = loss.MaskMeanSquaredError

        self.batch_size = SGD_p['minibatch_size']
        self.batch_num = args['epoch_size'] // self.batch_size
        self.epoch_num = args['epoch_num']

        self.log_train_dir = "log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
        self.log_test_dir = "log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/test'

        self.train_summary_writer = tf.summary.create_file_writer(self.log_train_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.log_test_dir)

    def build(self):
        self.init_state = tf.Variable(tf.ones([SGD_p['minibatch_size'], UNITS_SIZE]) * X_0,
                                      trainable=self.init_state_trainable)
        self.rnn_cell = rnn_cell.EIRNNCell(UNITS_SIZE, EI_RATIO)
        self.ei_rnn = keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)

        self.optimizer = keras.optimizers.SGD(learning_rate=SGD_p['lr'])



    def train(self):
        dg = DataGenerator(task_version=self.task_version, action='train')
        for epoch_i in range(self.epoch_num):
            for batch_i in range(self.batch_num):
                decs, masks, inputs, outputs = next(dg)

                with tf.GradientTape() as tape:

                    logits, states = self.ei_rnn(inputs, [self.init_state])
                    loss_value = self.loss_fun(outputs, tf.transpose(logits ,perm=[0, 2, 1]), masks)
                    loss_value += sum(self.ei_rnn.losses)


                grads = tape.gradient(loss_value, self.ei_rnn.trainable_weights)
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
                self.optimizer.apply_gradients(zip(grads, self.ei_rnn.trainable_weights))

            print('train loss:', loss_value.numpy())

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=epoch_i)
                w_rec_m = np.dot(np.multiply(self.rnn_cell.M_rec_m, self.rnn_cell.W_rec_plastic.numpy())\
                          + self.rnn_cell.W_fixed_m, self.rnn_cell.Dale_rec.numpy())
                cm_image = plot.plot_confusion_matrix(w_rec_m)

                tf.summary.image('M_rec', cm_image, step=epoch_i)




    def test(self):
        pass




# TODO:
#  Add terminate
#  Add output choice
#  Removed all weights below a threshold, wmin, after training.
#  Train it until its overall performance level of approximately 85%
#  Extract reaction time



"""
Test
from main import args
ei_rnn = SimpleEIRNN(args)
ei_rnn.build()
ei_rnn.train()

"""
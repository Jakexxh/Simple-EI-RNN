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
PERFORMANCE_LEVEL = 0.85

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

        self.log_train_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
        self.log_validation_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/validation'
        self.log_test_dir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/test'

        self.train_summary_writer = tf.summary.create_file_writer(self.log_train_dir)
        self.validation_summary_writer = tf.summary.create_file_writer(self.log_validation_dir)
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

            # Train
            #
            train_loss_all  = 0
            for batch_i in range(self.batch_num):
                decs, masks, inputs, outputs = next(dg)
                with tf.GradientTape() as tape:

                    logits, _ = self.ei_rnn(inputs, [self.init_state])
                    logits = tf.transpose(logits, perm=[0, 2, 1])
                    train_loss = self.loss_fun(outputs, logits, masks)
                    train_loss += sum(self.ei_rnn.losses)
                    train_loss_all += train_loss.numpy()

                grads = tape.gradient(train_loss, self.ei_rnn.trainable_weights)
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
                self.optimizer.apply_gradients(zip(grads, self.ei_rnn.trainable_weights))

            train_loss_all = train_loss_all / self.batch_num
            print('train loss:', train_loss_all)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_all, step=epoch_i)
                w_rec_m = np.dot(np.multiply(self.rnn_cell.M_rec_m, self.rnn_cell.W_rec_plastic.numpy())
                          + self.rnn_cell.W_fixed_m, self.rnn_cell.Dale_rec.numpy())
                cm_image = plot.plot_confusion_matrix(w_rec_m)

                tf.summary.image('M_rec', cm_image, step=epoch_i)


            # Validation
            #
            _, v_masks, v_inputs, v_outputs = dg.get_valid_test_datasets()
            v_logits, _ = self.ei_rnn(v_inputs, [self.init_state])

            acc_v_logits = v_logits.numpy()
            acc_v_outputs = tf.transpose(v_outputs, perm=[0, 2, 1]).numpy()
            validation_acc = self.get_accuracy(acc_v_logits, acc_v_outputs)

            v_logits = tf.transpose(v_logits, perm=[0, 2, 1])
            validation_loss = self.loss_fun(v_outputs, v_logits, v_masks)

            print('validation loss:', validation_loss.numpy())
            print('validation acc:', validation_acc)

            with self.validation_summary_writer.as_default():
                tf.summary.scalar('loss', validation_loss, step=epoch_i)
                tf.summary.scalar('acc', validation_acc, step=epoch_i)

            if validation_loss > PERFORMANCE_LEVEL:
                break

        # Test
        #
        self.test()

    def test(self, test_batch_num=100):
        dg = DataGenerator(task_version=self.task_version, action='train')  #todo: should be test for action
        psycollection = {'coh':[],'perc':[]}

        for batch_index in range(test_batch_num):
            descs, test_masks, test_inputs, test_outputs = dg.get_valid_test_datasets()

            test_logits, _ = self.ei_rnn(test_inputs, [self.init_state])

            acc_test_logits = test_logits.numpy()
            acc_test_outputs = tf.transpose(test_outputs, perm=[0, 2, 1]).numpy()
            test_acc = self.get_accuracy(acc_test_logits, acc_test_outputs)

            test_logits = tf.transpose(test_logits, perm=[0, 2, 1])
            test_loss = self.loss_fun(test_outputs, test_logits, test_masks)

            print('test loss:', test_loss.numpy())
            print('test acc:', test_acc)

            tmp_data = self.get_psychometric_data(descs, test_logits.numpy())

            psycollection['coh'] += tmp_data['coh']
            psycollection['perc'] += tmp_data['perc']

            with self.validation_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=batch_index)
                tf.summary.scalar('acc', test_acc, step=batch_index)

                curve_image = plot.plot_dots(psycollection['coh'], psycollection['perc'])

                tf.summary.image('psycollection', curve_image, step=batch_index)



    @staticmethod
    def get_accuracy(logits, outputs, collect_region=20):
        i,j,_ = np.shape(logits)
        element_num = i*j
        match_num = 0

        for i_index in range(i):
            for j_index in range(j-collect_region, j):
                logits_large_index = np.argmax(logits[i_index][j_index])
                outputs_large_index = np.argmax(outputs[i_index][j_index])
                if logits_large_index == outputs_large_index:
                    match_num += 1

        return match_num/element_num

    @staticmethod
    def get_psychometric_data(descs, logits, collect_region=20):
        data = {'coh':[],'perc':[]}

        for index in range(len(descs)):
            if descs[index]['choice'] == 0:
                data['coh'].append(-descs[index]['coh'])
            else:
                data['coh'].append(descs[index]['coh'])
            data['perc'].append(np.mean(logits[index][1][:-collect_region]))

        return data



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
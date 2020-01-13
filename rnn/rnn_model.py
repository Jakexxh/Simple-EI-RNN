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

np.set_printoptions(precision=5)

UNITS_SIZE = 100
EI_RATIO = 0.8
X_0 = 0.1
PERFORMANCE_LEVEL = 0.85
PERFORMANCE_CHECK_REGION = 3


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

        if args['model_date'] is None:
            self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.date = args['model_date']

        self.log_train_dir = "log/" + self.date + '_' + self.task_version + '/train'
        self.log_validation_dir = "log/" + self.date + '_' + self.task_version + '/validation'
        self.log_test_dir = "log/" + self.date + '_' + self.task_version + '/test'
        self.checkpoint_dir = "checkpoint/" + self.date + '/'

        self.train_summary_writer = tf.summary.create_file_writer(self.log_train_dir)
        self.validation_summary_writer = tf.summary.create_file_writer(self.log_validation_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.log_test_dir)

        self.ckpt = None
        self.ckpt_manager = None

    def build(self):
        self.init_state = tf.Variable(tf.ones([SGD_p['minibatch_size'], UNITS_SIZE]) * X_0,
                                      trainable=self.init_state_trainable)
        self.rnn_cell = rnn_cell.EIRNNCell(UNITS_SIZE, EI_RATIO, action='train')
        self.ei_rnn = keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)

        self.optimizer = keras.optimizers.SGD(learning_rate=SGD_p['lr'])

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.ei_rnn)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=3)

    def train(self):

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        dg = DataGenerator(task_version=self.task_version, action='train')
        validation_batch_num = self.batch_num // 10

        over_all_performace = []

        print('Start to train')
        print('#' * 20)

        for epoch_i in range(self.epoch_num):
            print('Epoch ' + str(epoch_i))

            # Train
            #
            train_loss_all = 0
            for batch_i in range(self.batch_num):
                decs, masks, inputs, outputs = next(dg)
                with tf.GradientTape() as tape:
                    logits, _ = self.ei_rnn(inputs, [self.init_state], training=True)
                    logits = tf.transpose(logits, perm=[0, 2, 1])
                    train_loss = self.loss_fun(outputs, logits, masks)
                    train_loss += sum(self.ei_rnn.losses)
                    train_loss_all += train_loss.numpy()

                grads = tape.gradient(train_loss, self.ei_rnn.trainable_weights)
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
                self.optimizer.apply_gradients(zip(grads, self.ei_rnn.trainable_weights))

                with self.train_summary_writer.as_default(): #TODO added to debug
                    cm_image = plot.plot_confusion_matrix(self.get_w_rec_m())
                    tf.summary.image('M_rec', cm_image, step=epoch_i)

            train_loss_all = train_loss_all / self.batch_num
            print('train loss:', train_loss_all)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_all, step=epoch_i)

            # Validation
            #
            validation_loss_all = 0
            validation_acc_all = 0

            for v_batch_i in range(validation_batch_num):
                _, v_masks, v_inputs, v_outputs = next(dg)#dg.get_valid_test_datasets()
                v_logits, _ = self.ei_rnn(v_inputs, [self.init_state])

                acc_v_logits = v_logits.numpy()
                acc_v_outputs = tf.transpose(v_outputs, perm=[0, 2, 1]).numpy()
                validation_acc = self.get_accuracy(acc_v_logits, acc_v_outputs, v_masks)
                validation_acc_all += validation_acc

                v_logits = tf.transpose(v_logits, perm=[0, 2, 1])
                validation_loss = self.loss_fun(v_outputs, v_logits, v_masks)
                validation_loss += sum(self.ei_rnn.losses)
                validation_loss_all += validation_loss.numpy()

            validation_loss_all = validation_loss_all / validation_batch_num
            validation_acc_all = validation_acc_all / validation_batch_num
            over_all_performace.append(validation_acc_all)

            print('validation loss:', validation_loss_all)
            print('validation acc:', validation_acc_all)

            with self.validation_summary_writer.as_default():
                tf.summary.scalar('loss', validation_loss_all, step=epoch_i)
                tf.summary.scalar('acc', validation_acc_all, step=epoch_i)

                cm_image = plot.plot_confusion_matrix(self.get_w_rec_m())
                tf.summary.image('M_rec', cm_image, step=epoch_i)

                win_image = plot.plot_confusion_matrix(funs.rectify(self.rnn_cell.W_in.numpy()[:, :int(UNITS_SIZE*EI_RATIO)]), False)
                tf.summary.image('M_in', win_image, step=epoch_i)

                wout_image = plot.plot_confusion_matrix(self.get_w_out_m()[:,:int(UNITS_SIZE*EI_RATIO)], False)
                tf.summary.image('M_out', wout_image, step=epoch_i)

            if epoch_i > PERFORMANCE_CHECK_REGION and \
                    np.mean(over_all_performace[-PERFORMANCE_CHECK_REGION:]) > PERFORMANCE_LEVEL:
                print('Overall performance level is satisfied, training is terminated\n')
                break

            # Save Model
            self.ckpt.step.assign_add(1)
            self.ckpt_manager.save()

            # self.reset_all_weights() # todo: may change
            # print('Remove all weights below ' + str(SGD_p['mini_w_threshold']))
            # print('\n')

        print('Training is done')
        print('Remove all weights below ' + str(SGD_p['mini_w_threshold']))
        print('#' * 20)
        print('\n')
        # Test
        #
        self.test()

    def test(self, test_batch_num=50):
        print('Start to test')
        print('#' * 20)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        dg = DataGenerator(task_version=self.task_version, action='test')  # todo: should be test for action
        psycollection = {'coh': [], 'perc': []}

        for batch_index in range(test_batch_num):
            descs, test_masks, test_inputs, test_outputs = dg.get_valid_test_datasets()

            test_logits, _ = self.ei_rnn(test_inputs, [self.init_state], training=False)

            acc_test_logits = test_logits.numpy()
            acc_test_outputs = tf.transpose(test_outputs, perm=[0, 2, 1]).numpy()
            test_acc = self.get_accuracy(acc_test_logits, acc_test_outputs, test_masks)

            test_logits = tf.transpose(test_logits, perm=[0, 2, 1])
            test_loss = self.loss_fun(test_outputs, test_logits, test_masks)
            test_loss += sum(self.ei_rnn.losses)

            print('test loss:', test_loss.numpy())
            print('test acc:', test_acc)

            tmp_data = self.get_psychometric_data(descs, test_logits.numpy())

            psycollection['coh'] += tmp_data['coh']
            psycollection['perc'] += tmp_data['perc']

            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=batch_index)
                tf.summary.scalar('acc', test_acc, step=batch_index)

                curve_image = plot.plot_dots(psycollection['coh'], psycollection['perc'])
                tf.summary.image('psycollection', curve_image, step=batch_index)

                cm_image = plot.plot_confusion_matrix(self.get_w_rec_m())
                tf.summary.image('M_rec', cm_image, step=batch_index)

                win_image = plot.plot_confusion_matrix(funs.rectify(self.rnn_cell.W_in.numpy()[:, :int(UNITS_SIZE*EI_RATIO)]), False)
                tf.summary.image('M_in', win_image, step=batch_index)

                wout_image = plot.plot_confusion_matrix(self.get_w_out_m()[:,:int(UNITS_SIZE*EI_RATIO)], False)
                tf.summary.image('M_out', wout_image, step=batch_index)

    def get_w_rec_m(self):
        return np.dot(self.rnn_cell.Dale_rec.numpy(),
                      np.multiply(self.rnn_cell.M_rec_m, funs.rectify(self.rnn_cell.W_rec_plastic.numpy()))
                                   + self.rnn_cell.W_fixed_m).T

    def get_w_out_m(self):
        return np.dot(self.rnn_cell.Dale_out.numpy(), funs.rectify(self.rnn_cell.W_out.numpy())).T

    def reset_all_weights(self):
        new_w = []
        for w in self.rnn_cell.get_weights():
            w[w < SGD_p['mini_w_threshold']] = 0.
            new_w.append(w)

        self.rnn_cell.set_weights(new_w)

    @staticmethod
    def get_accuracy(logits, outputs,masks): # todo: region may change for fd version
        batch_num, _ =np.shape(masks)
        means = 0.
        for index in range(batch_num):
            x = np.argmax(logits[index], axis=1)
            y = np.argmax(outputs[index], axis=1)
            delet_i, = np.where(masks[index] == 0)
            match = np.delete(np.equal(x,y),delet_i)
            means += np.mean(match)
        return means / batch_num

    @staticmethod
    def get_psychometric_data(descs, logits, collect_region=20):
        data = {'coh': [], 'perc': []}

        for index in range(len(descs)):
            if descs[index]['choice'] == 0:
                data['coh'].append(-descs[index]['coh'])
            else:
                data['coh'].append(descs[index]['coh'])
            choice_1_num = np.sum(logits[index][1][-collect_region:] - logits[index][0][-collect_region:] >= 0, axis=0)
            data['perc'].append(choice_1_num / collect_region)

        return data


# TODO:
#  Extract reaction time
#  No Cluster in Weights Matrix


"""
Test
from main import args
ei_rnn = SimpleEIRNN(args)
ei_rnn.build()
ei_rnn.train()

"""



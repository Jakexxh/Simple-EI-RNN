import tensorflow as tf
import scipy.stats as stats
import util.util_funs as funs
import random
import numpy as np
from main import SGD_p

np.set_printoptions(precision=5)
RT_FIX_T_MEAN = 200 #700
RT_REWARD_DELAY_T = 300
TRIAL_T = 2000  # TODO: may change
LOW_VALUE = 0.2
HIGH_VALUE = 1.2
REACTION_THRESHOLD = 1.0
CHOICE_MAP = [0, 1]
CATCH_TRIAL_RATIO = 0.2


class DataGenerator:

    def __init__(self, task_version='rt', action='train'):

        self.cohs = [0.032, 0.064, 0.128, 0.256, 0.512]
        self.fix_t = 0.
        self.ct_portion = 0.1  # self.p.ct_portion
        self.batch_size = SGD_p['minibatch_size']


        if task_version == 'rt':

            self.single_trial_fun = self.single_rt_train_trial

            if action == 'train':
                self.dt = SGD_p['train_t_step']
                self.trial_len = TRIAL_T // self.dt
                self.alpha = SGD_p['train_t_step'] / SGD_p['tau']
                self.fix_t = 100  # todo: may change
                self.step_flag = {
                    'fixation': (0, self.fix_t // self.dt),
                    'stimulus': (self.fix_t // self.dt, self.trial_len),
                    'decision': ((self.fix_t + RT_REWARD_DELAY_T) // self.dt, self.trial_len)}

            else:
                self.dt = SGD_p['test_t_step']
                self.trial_len = TRIAL_T // self.dt
                self.alpha = SGD_p['test_t_step'] / SGD_p['tau']
                self.fix_t = 300  # todo: may change
                self.step_flag = {
                    'fixation': (0, self.fix_t // self.dt),
                    'stimulus': (self.fix_t // self.dt, self.trial_len),
                    'decision': ((self.fix_t + RT_REWARD_DELAY_T) // self.dt, self.trial_len)}

        elif task_version == 'fd':

            self.single_trial_fun = self.single_fd_train_trial
            self.truncated_norm = self.get_truncated_norm(80, 1500, 330)

            if action == 'train':
                self.dt = SGD_p['train_t_step']
                self.trial_len = TRIAL_T // self.dt
                self.alpha = SGD_p['train_t_step'] / SGD_p['tau']
                self.fix_t = 100  # todo: may change
                self.step_flag = {
                    'fixation': (0, self.fix_t // self.dt)}
            else:
                self.dt = SGD_p['test_t_step']
                self.trial_len = TRIAL_T // self.dt
                self.alpha = SGD_p['test_t_step'] / SGD_p['tau']
                self.fix_t = 500  # todo: may change
                self.step_flag = {
                    'fixation': (0, self.fix_t // self.dt)}

        else:
            raise Exception('No task version: ' + task_version)

    def single_rt_train_trial(self):

        choice = 1 - random.choice(CHOICE_MAP)
        coh = random.choice(self.cohs)

        inputs = np.zeros((2, self.trial_len))
        outputs = np.zeros((2, self.trial_len))
        masks = np.zeros(self.trial_len)

        for step in range(self.trial_len):

            if step < self.step_flag['fixation'][1]:
                outputs[:, step] = LOW_VALUE
                masks[step] = 1

            if step >= self.step_flag['stimulus'][0]:
                inputs[choice][step] = funs.rectify(SGD_p['baseline_input'] +
                                                    self.stim_value(coh) + self.input_noise())
                inputs[1 - choice][step] = funs.rectify(SGD_p['baseline_input'] +
                                                        self.stim_value(-coh) + self.input_noise())

            if step >= self.step_flag['decision'][0]:
                outputs[choice][step] = HIGH_VALUE
                outputs[1 - choice][step] = LOW_VALUE
                masks[step] = 1

        return {'choice': choice, 'coh': coh},  masks, inputs.T, outputs

    def single_fd_train_trial(self):

        stim_t = self.truncated_norm.rvs(1)[0] // self.dt

        self.step_flag['stimulus'] = (self.step_flag['fixation'][1], self.step_flag['fixation'][1] + stim_t)
        self.step_flag['decision'] = (self.step_flag['fixation'][1] + stim_t, self.trial_len)

        choice = 1 - random.choice(CHOICE_MAP)
        coh = random.choice(self.cohs)

        inputs = np.zeros((2, self.trial_len))
        outputs = np.zeros((2, self.trial_len))
        masks = np.zeros(self.trial_len)

        for step in range(self.trial_len):

            if step < self.step_flag['fixation'][1]:
                outputs[:, step] = LOW_VALUE
                masks[step] = 1

            if self.step_flag['stimulus'][0] <= step < self.step_flag['stimulus'][1]:
                inputs[choice][step] = funs.rectify(SGD_p['baseline_input'] +
                                                    self.stim_value(coh) + self.input_noise())
                inputs[1 - choice][step] = funs.rectify(SGD_p['baseline_input'] +
                                                        self.stim_value(-coh) + self.input_noise())

            if step >= self.step_flag['decision'][0]:
                outputs[choice][step] = HIGH_VALUE
                outputs[1 - choice][step] = LOW_VALUE
                masks[step] = 1

        return {'choice': choice, 'coh': coh}, masks, inputs.T, outputs

    def single_catch_trial(self):
        inputs = np.zeros((2, self.trial_len))
        outputs = np.ones((2, self.trial_len)) * LOW_VALUE
        masks = np.ones(self.trial_len)

        return 'catch_trial', masks, inputs.T, outputs

    @staticmethod
    def get_truncated_norm(lower, upper, mean, scale=1.0):
        trunc_norm = stats.truncnorm(
            (lower - mean) / scale, (upper - mean) / scale, loc=mean, scale=scale)
        return trunc_norm

    def make_data(self):
        pass

    def stim_value(self, coh):
        return (1 + coh) / 2

    def input_noise(self):
        return (1 / self.alpha) * np.sqrt(2 * self.alpha * SGD_p['input_noise_std'] ** 2) * np.random.normal()

    def __iter__(self):
        return self

    def __next__(self):
        """
        generate trials and update iteration flag

        :return:
        """
        ctrial_num = int(self.batch_size * CATCH_TRIAL_RATIO)

        trials = []

        for num in range(self.batch_size):
            if num < ctrial_num:
                trials.append(self.single_catch_trial())
            else:
                trials.append(self.single_trial_fun())

        random.shuffle(trials)

        decs = []
        masks = []
        inputs_list = []
        outputs_list = []

        for row in trials:
            decs.append(row[0])
            masks.append(row[1])
            inputs_list.append(row[2])
            outputs_list.append(row[3])

        return decs, tf.convert_to_tensor(masks,dtype=np.float32), tf.convert_to_tensor(inputs_list,dtype=np.float32),\
               tf.convert_to_tensor(outputs_list,dtype=np.float32)

    def get_valid_test_datasets(self):
        trials = []
        for num in range(self.batch_size):
            trials.append(self.single_trial_fun())

        decs = []
        masks = []
        inputs_list = []
        outputs_list = []

        for row in trials:
            decs.append(row[0])
            masks.append(row[1])
            inputs_list.append(row[2])
            outputs_list.append(row[3])

        return decs, tf.convert_to_tensor(masks,dtype=np.float32), tf.convert_to_tensor(inputs_list,dtype=np.float32),\
               tf.convert_to_tensor(outputs_list,dtype=np.float32)


"""
Test

dg = DataGenerator()
a = next(dg)

for m, inputs, outputs in zip(a[1], a[2], a[2]):
    print(m)
    i = inputs.numpy()
    print(i)
    o = outputs.numpy()
    print(o)
    
"""


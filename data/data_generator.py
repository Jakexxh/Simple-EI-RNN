import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import util.util_funs as funs
import random
import numpy as np
from main import SGD_p
import sys

np.set_printoptions(precision=5)
RT_FIX_T_MEAN = 700
RT_REWARD_DELAY_T = 300
TRIAL_T = 3000  # TODO: may change
LOW_VALUE = 0.2
HIGH_VALUE = 1.0
CHOICE_MAP = [0, 1]
CATCH_TRIAL_RATIO = 0.2


class DataGenerator:

    def __init__(self, rnn_p):
        self.p = rnn_p

        self.cohs = [0.032, 0.064, 0.128, 0.256, 0.512]
        self.fix_t = 0.
        self.ct_portion = 0.1  # self.p.ct_portion
        self.dt = SGD_p['train_t_step']
        self.trial_len = int(TRIAL_T / self.dt)
        self.batch_size = SGD_p['minibatch_size']

        if self.p['task_version'] == 'rt':
            if self.p['action'] == 'train':
                self.alpha = SGD_p['train_t_step'] / SGD_p['tau']
                self.single_trial_fun = self.single_rt_train_trial
                self.fix_t = self.rt_mk_fix_t()
                self.step_flag = {
                    'fixation': (0, int(self.fix_t / self.dt)),
                    'stimulus': (int(self.fix_t / self.dt), self.trial_len),
                    'decision': (int((self.fix_t + RT_REWARD_DELAY_T) / self.dt), self.trial_len)}

            else:
                self.alpha = SGD_p['test_t_step'] / SGD_p['tau']
                pass

        elif self.p['task_version'] == 'fd':
            if self.p['action'] == 'train':
                self.single_trial_fun = self.single_rt_train_trial
            else:
                pass

        else:
            raise Exception('No task version: ' + self.p.task_version)

    def single_rt_train_trial(self):
        """

        :return: desc of trial, masks, inputs, outputs
        """
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

        return {'choice': choice, 'coh': coh}, masks, inputs, outputs

    def single_catch_trial(self):
        inputs = np.zeros((2, self.trial_len))
        outputs = np.ones((2, self.trial_len)) * LOW_VALUE
        masks = np.ones((1, self.trial_len))

        return 'catch_trial', masks, inputs, outputs

    def single_fd_train_trial(self):
        pass

    def rt_mk_fix_t(self, truncate=300):
        def exp_fun(): return np.random.exponential(RT_FIX_T_MEAN) // SGD_p['train_t_step'] * SGD_p['train_t_step']

        fix_t = exp_fun()

        while not (RT_FIX_T_MEAN - truncate <= fix_t <= RT_FIX_T_MEAN + truncate):
            fix_t = exp_fun()

        return fix_t

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

        return decs, masks, tf.data.Dataset.from_tensor_slices(inputs_list),\
               tf.data.Dataset.from_tensor_slices(outputs_list)


"""
Test

dg = DataGenerator({'task_version':'rt', 'action': 'train'})
a = next(dg)
print(a)
for elem in a[2]:
    print(elem.numpy()) #print inputs list
    
"""


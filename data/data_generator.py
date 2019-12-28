import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import util.util_funs as funs
import random
import numpy as np
from main import SGD_p

np.set_printoptions(precision=5)
RT_FIX_T_MEAN = 700
RT_REWARD_DELAY_T = 300
TRIAL_T = 3000  # TODO: may change
LOW_VALUE = 0.2
HIGH_VALUE = 1.2
CHOICE_MAP = [0, 1]


class DataGenerator:

    def __init__(self, rnn_p):
        self.p = rnn_p

        # self.choice =
        self.cohs = [0.032, 0.064, 0.128, 0.256, 0.512]
        self.fix_t = 0.
        self.ct_portion = 0.1  # self.p.ct_portion
        self.dt = SGD_p['train_t_step']
        self.trial_len = int(TRIAL_T / self.dt)

        if self.p.task_version == 'rt':
            if self.p.action == 'train':
                self.alpha = SGD_p['train_t_step'] / SGD_p['tau']
                self.trial_fun = self.single_rt_train_trial
                self.fix_t = self.rt_mk_fix_t()
                self.step_flag = {
                    'fixation': (0, int(self.fix_t / self.dt)),
                    'stimulus': (int(self.fix_t / self.dt), self.trial_len),
                    'decision': (int((self.fix_t + RT_REWARD_DELAY_T) / self.dt), self.trial_len)}

            else:
                self.alpha = SGD_p['test_t_step'] / SGD_p['tau']
                pass

        elif self.p.task_version == 'fd':
            if self.p.action == 'train':
                self.trial_fun = self.single_rt_train_trial
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
        masks = np.zeros((1, self.trial_len))

        for step in range(self.trial_len):

            if step < self.step_flag['fixation'][1]:
                outputs[step, :] = LOW_VALUE
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

        while not(RT_FIX_T_MEAN-truncate <= fix_t <= RT_FIX_T_MEAN+truncate):
            fix_t = exp_fun()

        return exp_fun

    def make_data(self):
        pass

    def stim_value(self, coh):
        return (1 + coh) / 2

    def input_noise(self):
        return (1 / self.alpha) + np.sqrt(2 * self.alpha * SGD_p['input_noise_std'] ** 2) * np.random.normal()

    def __iter__(self):
        return self

    def __next__(self):
        """
        generate trials and update iteration flag

        :return: desc, train trial, label trial
        """

        pass

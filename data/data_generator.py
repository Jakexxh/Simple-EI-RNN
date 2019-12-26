import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

np.set_printoptions(precision=5)


class DataGenerator:

    def __init__(self, rnn_p):
        self.p = rnn_p

        # self.choice =
        self.cohs = [0.032, 0.064, 0.128, 0.256, 0.512]

        if self.p.task_version == 'rt':
            if self.p.action == 'train':
                self.trail_fun = self.rt_train_trial
                self.fixation = 100
                self.stimulus = 800
                self.no_reward = 300
            else:
                pass

        elif self.p.task_version == 'fd':
            if self.p.action == 'train':
                self.trail_fun = self.fd_train_trial
            else:
                pass

        else:
            raise Exception('No task version: ' + self.p.task_version)

    def rt_train_trial(self):
        pass

    def fd_train_trial(self):
        pass

    def make_data(self):
        pass

    def stim_value(self, coh):
        return (1 + coh) / 2

    def __iter__(self):
        return self

    def __next__(self):
        """
        generate trials and update iteration flag

        :return: desc, train trial, train label
        """


        pass

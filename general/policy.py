# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:18
# @Author  : Enjoy Zhao
import random
import numpy as np
from scipy.stats import truncnorm
import abc
import tensorflow as tf

class Policy:

    #@abc.abstractmethod
    def act(self,**kwargs):

        return NotImplementedError()


class Greedy(Policy):


    def act(self,qvals):

        return np.argmax(qvals)




class EpsGreedy(Policy):
    """
    """
    def __init__(self,eps):

        self.eps=eps

    def act(self,qvals):

        if random.random()>self.eps:
            return np.argmax(qvals)
        return random.randrange(len(qvals))

class GaussianEpsGreedy(Policy):

    """

    """
    def __init__(self,eps_mean,eps_std):

        self.eps_mean=eps_mean

        self.eps_std=eps_std

    def act(self,qvals):

        eps=truncnorm.rvs((0-self.eps_mean)/self.eps_std, (1-self.eps_mean)/self.eps_std)

        if random.random() >eps:
            return np.argmax(qvals)
        return random.randrange(len(qvals))


class PassThrough(Policy):

    """

    """

    def act(self,action):
        return action
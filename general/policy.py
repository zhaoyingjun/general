# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:18
# @Author  : Enjoy Zhao
# @Describe ：在本文件中主要是定义策略基类和实现不同的策略方法类，已经实现的策略算法类包括Greedy 、EpsGreedy、GaussianEpsGreedy、PassThrough
import random
from scipy.stats import truncnorm
import abc
import tensorflow as tf

#定义Policy基类，使用@abc.abstractmethod装饰器来保证在实现该基类时必须实现的方法。对于策略基类，我们要求必须实现act方法。
class Policy:

    @abc.abstractmethod
    def act(self,**kwargs):
        return NotImplementedError()


#实现一个贪婪策略类，所谓贪婪算法就是每一步都选择收益最大的选项，使用tf.argmax来实现取最大值对应的index。
class Greedy(Policy):
    def act(self,qvals):
        return tf.argmax(qvals)

#实现一个EpsGreedy类，EpsGreedy是贪婪算法的改进，可以称为随机贪婪算法，在算法实现上加入一个随机选择开关，只有当随机数大于eps数是贪婪算法，否则就是随机选择。这个改进可以避免贪婪算法带来的局部最优解导致忽视全局最优解的情况。
class EpsGreedy(Policy):

    def __init__(self,eps):

        self.eps=eps

    def act(self,qvals):
        #当随机数大于eps时，使用贪婪算法，否则直接随机选择。
        if random.random()>self.eps:
            return tf.argmax(qvals)
        return random.randrange(len(qvals))
#实现一个高斯随机贪婪算法，是在随机贪婪算法的基础上对随机开关的取值eps进行了改进，将固定值eps修改为根据高斯分布来取值，这样可以更好的增加随机性，以具备更好的全局视野。
class GaussianEpsGreedy(Policy):

    def __init__(self,eps_mean,eps_std):

        self.eps_mean=eps_mean

        self.eps_std=eps_std

    def act(self,qvals):
        #构建一个高斯分布来取值，以增加全局随机性。
        eps=truncnorm.rvs((0-self.eps_mean)/self.eps_std, (1-self.eps_mean)/self.eps_std)
        if random.random() >eps:
            return tf.argmax(qvals)
        return random.randrange(len(qvals))

#实现一个透传策略，其实就是把输入原封不动的输出
class PassThrough(Policy):

    def act(self,action):
        return action
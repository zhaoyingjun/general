# -*- coding: utf-8 -*-
# @Time    : 2025-07-08 03:54
# @Author  : Enjoy Zhao
# @Describe ：在本文件中主要是定义策略基类和实现不同的策略方法类，已经实现的策略算法类包括Greedy 、EpsGreedy、GaussianEpsGreedy、PassThrough（PyTorch版本）

import random
from scipy.stats import truncnorm
import abc
import torch
import numpy as np
# 定义Policy基类
class Policy(abc.ABC):
    @abc.abstractmethod
    def act(self, **kwargs):
        raise NotImplementedError()

# 贪婪策略：每次选择最大Q值的动作
class Greedy(Policy):
    def act(self, qvals):
        # 支持 numpy 数组或 torch tensor
        if isinstance(qvals, torch.Tensor):
            return torch.argmax(qvals).item()
        else:
            return int(np.argmax(qvals))

# EpsGreedy策略：以概率eps随机，1-eps贪婪
class EpsGreedy(Policy):
    def __init__(self, eps):
        self.eps = eps

    def act(self, qvals):
        if random.random() > self.eps:
            if isinstance(qvals, torch.Tensor):
                return torch.argmax(qvals).item()
            else:
                return int(np.argmax(qvals))
       
        return random.randrange(len(qvals))

# GaussianEpsGreedy策略：eps服从截断高斯分布
class GaussianEpsGreedy(Policy):
    def __init__(self, eps_mean, eps_std):
        self.eps_mean = eps_mean
        self.eps_std = eps_std

    def act(self, qvals):
        eps = truncnorm.rvs((0-self.eps_mean)/self.eps_std, (1-self.eps_mean)/self.eps_std, loc=self.eps_mean, scale=self.eps_std)
        if random.random() > eps:
            if isinstance(qvals, torch.Tensor):
                return torch.argmax(qvals).item()
            else:
                return int(np.argmax(qvals))
        return random.randrange(len(qvals))

# 透传策略，直接返回输入
class PassThrough(Policy):
    def act(self, action):
        return action
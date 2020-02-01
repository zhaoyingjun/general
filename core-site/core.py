# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 20:54
# @Author  : Enjoy Zhao
"""
Abstract base class for exception and client
在本文件中是定义exception和client的抽象基类，注意抽象基类并不能直接使用，需要根据抽象基类的定义具体实现之后才能使用
"""
class GrEexception(Exception):
    """
    Basic exception for errors raised by General.
    抛出的异常方法基于本抽象基类来实现
    """

class Agent:
    """
    Abstract base class for all implemented agents.
    agent是指在强化学习中的训练的智能体，在本抽象基类中包括agent所有需要具备的属性.
    """
    def save(self,filename,overwrite=False):
        """
        Saves the model parameters to the specified file.
        save属性将智能体中的预测模型参数保存到模型文件中
        """
        raise NotImplementedError()

    def act(self,sate,instance=0):
        """
        Returns the action to be taken given a state.
        act属性是需要实现根据输入的环境状态输出执行动作指令
        """
        raise NotImplementedError()

    def push(self,transition,instance=0):
        """
        Stores the transition in memory.
        push 属性是将智能体与环境交互的经验存储到记忆器中
        """
        raise NotImplementedError()

    def train(self,step):
        """
        Trains the agent for one step.
        train属性是完成对智能体神经网络的训练。
        """
        raise NotImplementedError()






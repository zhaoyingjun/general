# -*- coding: utf-8 -*-
# @Time    : 2025-07-08 03:57
# @Author  : Enjoy Zhao

import torch
import torch.nn as nn
import general_pytorch as gr
import os
import configparser

if not os.path.exists("model_dir"):
    os.makedirs("model_dir")

class client(object):
    def __init__(self, dense_num=None, cell_num=None, activation=None, action_space=0,
                 train_steps=3000, run_steps=1000, dummy_env=None, model_name=None, algorithm_type=None):

        self.dense_num = dense_num
        self.cell_num = cell_num
        self.activation = activation
        self.train_steps = train_steps
        self.run_steps = run_steps
        self.dummy_env = dummy_env
        self.model_name = model_name
        self.file_path = "model_dir/" + model_name + ".pt"
        self.config_path = "model_dir/" + model_name + ".cfg"
        self.algorithm_type = algorithm_type
        self.action_space = action_space
        self.ep_reward = 0
        self.ep_step = 0
        self.done = False

    def _get_activation(self, act):
        # 支持常见的激活函数字符串
        mapping = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return mapping.get(act, nn.ReLU())

    def create_model(self):
        layers = []
        input_dim = self.dummy_env.observation_space.shape[0]
        layers.append(nn.Linear(input_dim, 16))
        layers.append(nn.ReLU())
        for i in range(self.dense_num):
            layers.append(nn.Linear(16 if i == 0 else self.cell_num, self.cell_num))
            layers.append(self._get_activation(self.activation))
        model = nn.Sequential(*layers)
        return model

    def plot_rewards(self, episode_rewards, episode_steps, done=False):
        self.ep_reward = episode_rewards[0][-1]
        self.ep_step = episode_steps[0][-1]
        self.done = done

    def create_agent(self):
        model = self.create_model()
        if self.action_space == 0:
            self.action_space = self.dummy_env.action_space.n
        # 选择算法类型
        if self.algorithm_type == 'dqn':
            agent = gr.DQN(model, actions=self.action_space, nsteps=2)
        elif self.algorithm_type == 'ddpg':
            agent = gr.DDPG(model, actions=self.action_space, nsteps=2)
        elif self.algorithm_type == 'ppo':
            agent = gr.PPO(model, actions=self.action_space, nsteps=2)
        else:
            raise ValueError("Unsupported algorithm_type")
        return agent

    def train(self):
        agent = self.create_agent()
        # 加载已存在的权重
        if os.path.exists(self.file_path):
            agent.model.load_state_dict(torch.load(self.file_path))
        # 训练
        sim = gr.Trainer(self.dummy_env, agent)
        sim.train(max_steps=self.train_steps, visualize=True, plot=self.plot_rewards)
        # 保存模型权重
        torch.save(agent.model.state_dict(), self.file_path)
        # 保存配置
        cf = configparser.ConfigParser(allow_no_value=True)
        cf.add_section('ints')
        cf.add_section('strings')
        cf.set('ints', 'dense_num', "%s" % self.dense_num)
        cf.set('ints', 'cell_num', "%s" % self.cell_num)
        cf.set('ints', 'action_space', "%s" % self.action_space)
        cf.set('strings', 'activation', "%s" % self.activation)
        cf.set('strings', 'algorithm_type', "%s" % self.algorithm_type)
        with open(self.config_path, 'w') as f:
            cf.write(f)

    def run_model(self):
        agent = self.create_agent()
        # 加载模型权重
        agent.model.load_state_dict(torch.load(self.file_path))
        sim = gr.Trainer(self.dummy_env, agent)
        sim.test(max_steps=self.run_steps, plot=self.plot_rewards)
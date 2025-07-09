# -*- coding: utf-8 -*-
# @Time    : 2025-07-08 03:58
# @Author  : Enjoy Zhao

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("wxagg")  # 
import matplotlib.pyplot as plt
import gym
import general_pytorch as gr
import os

# 初始化gym环境
create_env = lambda: gym.make('CartPole-v1',render_mode='human').unwrapped
dummy_env = create_env()

if not os.path.exists("model_dir"):
    os.makedirs("model_dir")

def create_model():
    # PyTorch神经网络，三层，每层16单元
    input_dim = dummy_env.observation_space.shape[0]
    model = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
    )
    return model

def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001)

def train():
    model = create_model()
    agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=1000)
    checkpoint_path = "model_dir/dqn.pt"
    if os.path.exists(checkpoint_path):
        agent.model.load_state_dict(torch.load(checkpoint_path))
    tra = gr.Trainer(dummy_env, agent)
    tra.train(max_steps=1000, visualize=True, plot=plot_rewards)
    torch.save(agent.model.state_dict(), checkpoint_path)

def test():
    model = create_model()
    agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=2)
    checkpoint_path = "model_dir/dqn.pt"
    if os.path.exists(checkpoint_path):
        agent.model.load_state_dict(torch.load(checkpoint_path))
    tra = gr.Trainer(dummy_env, agent)
    tra.test(max_steps=300)

if __name__ == '__main__':
    print("请准确输入train或者test")
    mode = input()
    if mode == "train":
        train()
    elif mode == "test":
        test()
    else:
        print("请重新执行程序并准确输入train或者test")
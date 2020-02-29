# -*- coding: utf-8 -*-
# @Time    : 2020-03-01 00:06
# @Author  : Enjoy Zhao

import tensorflow as tf
import matplotlib
matplotlib.use("macOSX")#在使用macOSX系统时需要该行
import matplotlib.pyplot as plt
import gym
import general as gr
import os
#初始化gym环境，使用CartPole-v0环境，就是托扁担游戏
create_env = lambda: gym.make('CartPole-v0').unwrapped
dummy_env = create_env()
if not os.path.exists("model_dir"):
      os.makedirs("model_dir")
def create_model():
      # 我们使用tensorflow2.0中的高阶API keras定义一个全连接神经网络，用于学习预测。一共三层，每层16个神经元，激活函数使用relu。
   model = tf.keras.Sequential([
         tf.keras.layers.Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
         tf.keras.layers.Dense(16, activation='relu'),
         tf.keras.layers.Dense(16, activation='relu'),
      ])
   return model
#定义反馈画图函数，这是为了能够图形化展示训练过程中rewards的变化走势，rewards是用来反馈对智能体的行为的评价。
def plot_rewards(episode_rewards, episode_steps, done=False):
      plt.clf()
      plt.xlabel('Step')
      plt.ylabel('Reward')
      for ed, steps in zip(episode_rewards, episode_steps):
         plt.plot(steps, ed)
      plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated
def train():
   #初始化神经网络模型
   model=create_model()
   #将定义好的网络作为参数传入general框架的API中，构成一个完成DQN 智能体，用于接下来的强化学习训练。
   agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=2)
   cpkt=tf.io.gfile.listdir("model_dir")
   if cpkt:
      agent.model.load_weights("model_dir/dqn.h5")
   #将智能体和gym环境放入训练器中开始训练深度神经网络模型
   tra = gr.Trainer(dummy_env, agent)
   tra.train(max_steps=3000, visualize=True, plot=plot_rewards)
   agent.save(filename='model_dir/dqn.h5',overwrite=True,save_format='h5')
def test():
   #初始化神经网络模型
    model=create_model()
    #将定义好的网络作为参数传入general框架的API中，构建一个含有DQN神经网络的智能体。
    agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=2)
    #将之前训练的模型参数导入的新初始化的神经网络中
    agent.model.load_weights("model_dir/dqn.h5")
    #将智能体和gym环境放入训练器中开始测试模型的效果
    tra = gr.Trainer(create_env, agent)
    tra.test(max_steps=1000)
if __name__ == '__main__':
   print("请准确输入train或者test")
   mode=input()

   if mode=="train":
      train()
   elif mode=="test":
      test()
   else:
      print("请重新执行程序并准确输入train或者test")


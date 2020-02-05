# -*- coding: utf-8 -*-
# @Time    : 2020-02-01 20:40
# @Author  : Enjoy Zhao

import tensorflow as tf
#import matplotlib
#matplotlib.use("macOSX")
import general as gr
import os
if not os.path.exists("model_dir"):
      os.makedirs("model_dir")

class client(object):

    def __init__(self,dense_num=None,cell_num=None,activation=None,train_steps=3000,run_steps=None,dummy_env=None,model_name=None,algorithm_type=None):

        self.dense_num=dense_num
        self.cell_num=cell_num
        self.activation=activation
        self.train_steps=train_steps
        self.dummy_env=dummy_env
        self.model_name=model_name
        self.file_path="model_dir/"+model_name
        self.algorithm_type=algorithm_type
        self.run_steps=run_steps
        self.ep_reward=0
        self.ep_step=0
        self.done=False

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=self.dummy_env.observation_space.shape))
        for i in range(self.dense_num):
            model.add(tf.keras.layers.Dense(self.cell_num,activation=self.activation ))
        return model


    def plot_rewards(self,episode_rewards, episode_steps, done=False):

        self.ep_reward=episode_rewards[0][-1]

        self.ep_step=episode_steps[0][-1]
        self.done=done




    def create_agent(self):
        model = self.create_model()
        if self.algorithm_type=='dqn':
            agent = gr.DQN(model, actions=2, nsteps=2)
        elif self.algorithm_type=='ddpg':
            agent = gr.DDPG(model, actions=self.dummy_env.action_space.n, nsteps=2)

        elif self.algorithm_type=='ppo':
            agent = gr.PPO(model, actions=self.dummy_env.action_space.n, nsteps=2)

        return agent


    def train(self):

        # 将定义好的网络作为参数传入框架的API中，构成一个完成智能体，用于接下来的强化学习训练。
        agent =self.create_agent()
        cpkt = tf.io.gfile.listdir("model_dir")
        if cpkt:
            agent.model.load_weights(self.file_path)
        # 使用huskarl框架的simulation来创建一个训练模拟器，在模拟器中进行训练。
        sim = gr.Simulation(self.dummy_env, agent)
        sim.train(max_steps=self.train_steps, visualize=True, plot=self.plot_rewards)
        agent.model.save_weights(filepath=self.file_path, overwrite=True, save_format='h5')

    def run_model(self):
        model = self.create_model()
        agent = gr.DQN(model, actions=self.dummy_env.action_space.n, nsteps=2)
        agent.model.load_weights(filepath=self.file_path)
        sim = gr.Simulation(self.dummy_env, agent)
        sim.test(max_steps=self.run_steps)




# if __name__ == '__main__':
# 	dummy_env=lambda: gym.make('CartPole-v0').unwrapped
# 	dummy_env=dummy_env()

    #s=client(dense_num=3,cell_num=16,activation='relu',train_steps=3000,dummy_env=dummy_env,model_name='dqn',algorithm_type='dqn')
    #s.train()
# 	print("请准确输入gym或者url：")
# 	env = input()
# 	if env == "gym":
# 		print("系统默认时间用CartPole小游戏为您演示")
# 		create_env = lambda: gym.make('CartPole-v0').unwrapped
# 		dummy_env = create_env()
#
# 	elif env == "url":
# 		print("如果您使用本项目自带的外部环境示例，需要先执行python3 apitest.py")
# 		print("============================================================")
# 		print("请输入外部环境url：")
# 		url=input()
# 		print("请输入外部环境状态向量：")
# 		s_vectory=input()
# 		s_vectory=np.array(s_vectory.split( ','))
# 		print(s_vectory)
# 		print(type(s_vectory))
#
# 		print("请输入外部环境复位状态：")
#
# 		reset_state=input()
# 		reset_state=reset_state.split(',')
# 		print(reset_state)
# 		print(type(reset_state))
#
# 		#create_env = Proxy(url='http://127.0.0.1:5000/', observation_space=s_vectory,
# 						  # init_state=reset_state)
# 		create_env = Proxy(url='http://127.0.0.1:5000/', observation_space=np.array([0, 0, 0, 0]),
# 						   init_state=[-0.00924334, -0.01935845, 0.01062945, 0.04368442])
#
# 		dummy_env = create_env
#
# 	else:
# 		print("请重新执行程序并准确输入gym或者test")
#
# 	# Setup gym environment
# 	#create_env = lambda: gym.make('CartPole-v0').unwrapped
# 	#create_env=Proxy(url='http://127.0.0.1:5000/',observation_space=np.array([0,0,0,0]),init_state=[-0.00924334, -0.01935845,  0.01062945,  0.04368442])
# 	#dummy_env = create_env
#
# 	# Build a simple neural network with 3 fully connected layers as our model
# 	model = Sequential([
# 		Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
# 		Dense(16, activation='relu'),
# 		Dense(16, activation='relu'),
# 	])
#
# 	# Create Deep Q-Learning Network agent
# 	agent = gr.DQN(model, actions=2, nsteps=2)
#
# 	def plot_rewards(episode_rewards, episode_steps, done=False):
# 		plt.clf()
# 		plt.xlabel('Step')
# 		plt.ylabel('Reward')
# 		for ed, steps in zip(episode_rewards, episode_steps):
# 			plt.plot(steps, ed)
# 		plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated
#
# 	# Create simulation, train and then test
# 	sim = gr.Simulation(dummy_env, agent)
# 	sim.train(max_steps=3000, visualize=True, plot=plot_rewards)
# 	sim.test(max_steps=1000)

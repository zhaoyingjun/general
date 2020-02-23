# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:18
# @Author  : Enjoy Zhao
# @Describe ：本文件是定义了一个训练器方法类，在训练器中定义了训练函数，包括单实例和多实例训练,强化学习的训练是在训练模拟器中完成的。
import multiprocessing as mp
from collections import namedtuple
import tensorflow as tf
import cloudpickle
from general.core import GrEexception
from general.memory import Transition

#定义全局变量RewardState
RewardState=namedtuple('RewardState',['reward','state'])
#定义Trainer方法类，实现在训练器的训练功能
class Trainer:

    #定义初始化方法，将create_env、agent、mapping进行初始化
    def __init__(self,create_env,agent,mapping=None):

        self.create_env=create_env
        self.agent=agent
        self.mapping=mapping
     #定义train方法，根据配置对智能体进行训练
    def train(self,max_steps=1000,instances=1,visualize=False,plot=None,max_subprocesses=0):

        #将智能体设置为训练状态
        self.agent.training=True
        #如果是单进程训练，则调用_sp_train方法,如果是多进程训练则调用_mp_train方法
        if max_subprocesses==0:
            self._sp_train(max_steps,instances,visualize,plot)
        elif max_subprocesses is None or max_subprocesses>0:

            self._mp_train(max_steps,instances,visualize,plot,max_subprocesses)
        else:
            raise GrEexception(f"Invalid max_subprocesses setting: {max_subprocesses}")
    #定义单进程训练方法
    def _sp_train(self, max_steps, instances, visualize=False, plot=None):

        """
        :param max_steps:最大训练步数
        :param instances:训练智能体的数量
        :param visualize:配置是否图形可视化，针对与gym适用
        :param plot:画图函数，对训练步数和rewards进行画图
        """
        #根据设置的instances的数量也就是智能体的数量，分别初始化reward、step、envs、states，用于训练过程的图形化展示
        episode_reward_sequences=[[0] for _ in range(instances)]
        episode_step_sequences=[[0] for _ in range(instances)]
        episode_rewards=[0]*instances

        envs=[self.create_env for _ in range(instances)]
        states=[env.reset() for env in envs]

        #训练步数在最大步数范围内开始循环训练
        for step in range(max_steps):
            #根据智能体的数量和是否进行图形可视化，进行环境可视化，这里只适用于gym环境
            for i in range(instances):
                if visualize:envs[i].render()
                #将预测得到的action从Tensor转换为数值
                action = tf.keras.backend.eval(self.agent.act(states[i], i))
                #将预测得到的action输入给环境，获得环境反馈的下一个状态、reward、和是否结束的标记
                next_state,reward,done, _=envs[i].step(action=action)
                #将环境返回的数据、action作为一条记录保存到记忆存储器中
                self.agent.push(Transition(states[i],action,reward,None if done else next_state),i)
                #将reward进行累加
                episode_rewards[i]+=reward
                #如果环境给予结束的状态则将episode_rewards、训练步数给保存，episode_rewards清零后进行图形展示，如果不是结束状态则将状态更新为下一个状态。
                if done:
                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    episode_rewards[i]=0
                    if plot:
                        plot(episode_reward_sequences,episode_step_sequences)
                    states[i]=envs[i].reset()
                else:
                    states[i]=next_state
            #训练智能体,完成一步的训练
            self.agent.train(step)
        # 最后图形化展示整个训练过程的reward

        if plot:plot(episode_reward_sequences,episode_step_sequences,done=True)
    #定义多进程训练方法，涉及到多进程调度、通信等过程，相对实现起来比较复杂。
    def _mp_train(self,max_steps,instances,visualize,plot,max_subprocesses):

        #如果最大子进程数量没有设置，则根据cpu的数量来作为最大子进程数量
        if max_subprocesses is None:

            max_subprocesses=mp.cpu_count()
        #智能体的数量和最大子进程的数量取最小值
        nprocesses=min(instances,max_subprocesses)

        #计算每一个进程中智能体的数量
        instances_per_process=[instances//nprocesses] * nprocesses
        #计算左侧溢出的智能体，也就是排队的智能体
        leftover=instances%nprocesses
        #如果有智能体排队，则均匀的分配排队智能体的数量
        if leftover>0:
            for i in range(leftover):
                instances_per_process[i]+=1

        #计算智能体的id，通过枚举的方式逐个获得智能体的id
        instance_ids=[list(range(i,instances,nprocesses))[:ipp] for i ,ipp in enumerate(instances_per_process)]
        #初始化一个pipes和processes数组，用于存储等待状态或者中间状态的数据
        pipes=[]
        processes=[]
        #逐个将智能体实例调度到不同的子进程中，等待训练。
        for i in range(nprocesses):
            child_pipes=[]
            #以下过程是将不同的智能体装入到子进程中的过程。
            for j in range(instances_per_process[i]):
                #获得父进程和子进程的pipe
                parent,child=mp.Pipe()
                #将进程和子进程的pipe保存到各自的数组中
                pipes.append(parent)
                child_pipes.append(child)
            #将多个智能体以及训练参数装入到子进程中,这里的训练函数使用的是专门为多进程训练编写的函数，具体过程见_train
            pargs=(cloudpickle.dumps(self.create_env),instance_ids[i],max_steps,child_pipes,visualize)
            processes.append(mp.Process(target=_train,args=pargs))

        print(f"Starting {nprocesses} process(es) for {instances} environment instance(s)... {instance_ids}")
        #启动所有的进程开始训练
        for p in processes:p.start()
        # 根据设置的instances的数量也就是智能体的数量，分别初始化reward、step、envs、states，用于训练过程的图形化展示
        episode_reward_sequences = [[] for _ in range(instances)]
        episode_step_sequences = [[] for _ in range(instances)]
        episode_rewards = [0] * instances

        #初始化rss和last_actions
        rss=[None]*instances

        last_actions=[None]*instances


        #开始在最大训练步数范围进行循环训练
        for step in range(max_steps):
            #初始化完成训练的数量,全部初始化为未完成训练
            step_done=[False]*instances
            #如果没有全部完成训练，则需要进行等待，直到全部完成训练。
            while sum(step_done)<instances:
                #获得需要等待完成完成训练的子进程
                awaiting_pipes=[p for iid,p in enumerate(pipes) if step_done[iid]==0]
                #获得已经完成训练的awaiting_pipes
                ready_pipes=mp.connection.wait(awaiting_pipes,timeout=None)
                #获得完成训练的子进程的idex
                pipe_indexes=[pipes.index(rp) for rp in ready_pipes]
                #对完成训练的子进程idex进行排序
                pipe_indexes.sort()
                #将完成训练的进程中环境状态、action、reward、下一步的状态存储到智能体记忆存储器中
                for iid in pipe_indexes:
                    #从进程间管道中接收信息
                    rs =pipes[iid].recv()
                    if rss[iid] is not None:

                        exp=Transition(rss[iid].state,last_actions[iid],rs.reward,rs.state)

                        self.agent.push(exp,iid)
                        #将训练结束状态标记为True
                        step_done[iid]=True
                    rss[iid]=rs
                    #如果环境给予结束的状态则将episode_rewards、训练步数给保存，episode_rewards清零后进行图形展示
                    if rs.state is None:
                        rss[iid]=None
                        episode_reward_sequences[iid].append(episode_rewards[iid])

                        episode_step_sequences[iid].append(step)

                        episode_rewards[iid]=0
                        if plot:
                            plot(episode_reward_sequences,episode_step_sequences)
                    #如果不是结束状态则获得智能体的执行动作，并更新last_action，同时通过管道将action发送出去。
                    else:
                        action=self.agent.act(rs.state,iid)
                        last_actions[iid]=action

                        try:
                            pipes[iid].send(action)
                        except BrokenPipeError as bpe:
                            if step <(max_steps-1):raise bpe
                        #如果在管道中还能接收到reward则将episode_rewards[iid]进行更新
                        if rs.reward:episode_rewards[iid]+=rs.reward
            #训练智能体，完成一步训练
            self.agent.train(step)
        #最后图形化展示整个训练过程的reward
        if plot:plot(episode_reward_sequences, episode_step_sequences, done=True)
    #定义测试方法，对完成训练的智能体进行测试
    def test(self,max_steps,visualize=True):
       #将智能体的网络设置测试状态
        self.agent.training=False
        #创建环境并初始化
        env=self.create_env()

        state=env.reset()
        #在最大测试步数范围测试智能体与环境的交互
        for step in range(max_steps):
            if visualize:env.render()
            action=tf.keras.backend.eval(self.agent.act(state))
            next_state,reward,done,_=env.step(action)
            state=env.reset() if done else  next_state
#定义多进程训练方法，其中涉及到来训练过程的进程间通信
def _train(create_env,instance_ids,max_steps,pipes,visualize):

    #根据智能体的数量初始化pipes、actions、envs
    pipes={iid:p for iid,p in zip(instance_ids,pipes)}

    actions={iid:None for iid in instance_ids}

    create_env=cloudpickle.loads(create_env)

    envs={iid:create_env for iid in instance_ids}

    #获得各个智能体对应环境的初始状态，并通过pipes管道进行进程间通信
    for iid in instance_ids:
        state=envs[iid].reset()

        pipes[iid].send(RewardState(None,state))
    #在最大训练步数的范围，开始进行循环训练
    for step in range(max_steps):
        for iid in instance_ids:
            #从管道中接收数据，更新actions
            actions[iid]=pipes[iid].recv()
            #如果需要图像可视化，则展示环境的可视化，适用于gym
            if visualize: envs[iid].render()
            #将获得的action输入环境中获得next_state,reward,done
            next_state,reward,done,_=envs[iid].step(actions[iid])
            #将RewardState通过管道发送出去
            pipes[iid].send(RewardState(reward,None if done else  next_state))

            #如果环境终止来，则将环境初始化，并发送RewardState
            if done:

                state=envs[iid].reset()

                pipes[iid].send(RewardState(None,state))













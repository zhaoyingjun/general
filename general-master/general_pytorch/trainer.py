# -*- coding: utf-8 -*-
# @Time    : 2025-07-08 03:55
# @Author  : Enjoy Zhao
# @Describe ：本文件定义了一个训练器方法类，支持单实例和多实例的强化学习训练（PyTorch版本）。

import multiprocessing as mp
from collections import namedtuple
import cloudpickle
from general_pytorch.core import GrEexception
from general_pytorch.memory import Transition

# 定义全局变量RewardState
RewardState = namedtuple('RewardState', ['reward', 'state'])

class Trainer:
    def __init__(self, create_env, agent, mapping=None):
        self.create_env = create_env
        self.agent = agent
        self.mapping = mapping

    def train(self, max_steps=1000, instances=1, visualize=False, plot=None, max_subprocesses=0):
        self.agent.training = True
        if max_subprocesses == 0:
            self._sp_train(max_steps, instances, visualize, plot)
        elif max_subprocesses is None or max_subprocesses > 0:
            self._mp_train(max_steps, instances, visualize, plot, max_subprocesses)
        else:
            raise GrEexception(f"Invalid max_subprocesses setting: {max_subprocesses}")

    def _sp_train(self, max_steps, instances, visualize=True, plot=None):
        episode_reward_sequences = [[0] for _ in range(instances)]
        episode_step_sequences = [[0] for _ in range(instances)]
        episode_rewards = [0] * instances

        envs = [self.create_env for _ in range(instances)]
        states = [env.reset() for env in envs]

        

        for step in range(max_steps):
            for i in range(instances):
                envs[i].render()
                if visualize:
                    envs[i].render()
                # 预测得到的action转为数值

                action = self.agent.act(states[i][0], i)
                action= max(0, min(action,  envs[i].action_space.n-1))
                # 若action为tensor，取item
                if hasattr(action, "item"):
                    action = action.item()
                next_state, reward, done, _ ,_= envs[i].step(action)
    

                #print(reward)
                self.agent.push(Transition(states[i], action, reward, None if done else next_state), i)
                
                episode_rewards[i] += reward
                episode_reward_sequences[i].append(episode_rewards[i])
                episode_step_sequences[i].append(step)
                if plot:
                    plot(episode_reward_sequences, episode_step_sequences)
                if done:
                    episode_rewards[i] = 0
                    states[i] = envs[i].reset()
                else:
                    states[i] = [next_state]
            # 训练一步

            self.agent.train(step)
            
        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def _mp_train(self, max_steps, instances, visualize, plot, max_subprocesses):
        if max_subprocesses is None:
            max_subprocesses = mp.cpu_count()
        nprocesses = min(instances, max_subprocesses)
        instances_per_process = [instances // nprocesses] * nprocesses
        leftover = instances % nprocesses
        if leftover > 0:
            for i in range(leftover):
                instances_per_process[i] += 1

        instance_ids = [list(range(i, instances, nprocesses))[:ipp] for i, ipp in enumerate(instances_per_process)]
        pipes = []
        processes = []
        for i in range(nprocesses):
            child_pipes = []
            for j in range(instances_per_process[i]):
                parent, child = mp.Pipe()
                pipes.append(parent)
                child_pipes.append(child)
            pargs = (cloudpickle.dumps(self.create_env), instance_ids[i], max_steps, child_pipes, visualize)
            processes.append(mp.Process(target=_train, args=pargs))

        print(f"Starting {nprocesses} process(es) for {instances} environment instance(s)... {instance_ids}")
        for p in processes:
            p.start()
        episode_reward_sequences = [[] for _ in range(instances)]
        episode_step_sequences = [[] for _ in range(instances)]
        episode_rewards = [0] * instances
        rss = [None] * instances
        last_actions = [None] * instances

        for step in range(max_steps):
            step_done = [False] * instances
            while sum(step_done) < instances:
                awaiting_pipes = [p for iid, p in enumerate(pipes) if not step_done[iid]]
                ready_pipes = mp.connection.wait(awaiting_pipes, timeout=None)
                pipe_indexes = [pipes.index(rp) for rp in ready_pipes]
                pipe_indexes.sort()
                for iid in pipe_indexes:
                    rs = pipes[iid].recv()
                    if rss[iid] is not None:
                        exp = Transition(rss[iid].state, last_actions[iid], rs.reward, rs.state)
                        self.agent.push(exp, iid)
                        step_done[iid] = True
                    rss[iid] = rs
                    if rs.state is None:
                        rss[iid] = None
                        episode_reward_sequences[iid].append(episode_rewards[iid])
                        episode_step_sequences[iid].append(step)
                        episode_rewards[iid] = 0
                        if plot:
                            plot(episode_reward_sequences, episode_step_sequences)
                    else:
                        action = self.agent.act(rs.state, iid)
                        if hasattr(action, "item"):
                            action = action.item()
                        last_actions[iid] = action
                        try:
                            pipes[iid].send(action)
                        except BrokenPipeError as bpe:
                            if step < (max_steps - 1):
                                raise bpe
                        if rs.reward:
                            episode_rewards[iid] += rs.reward
            # 训练一步
            self.agent.train(step)
        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def test(self, max_steps, instances=1, visualize=False, plot=None):
        episode_reward_sequences = [[0] for _ in range(instances)]
        episode_step_sequences = [[0] for _ in range(instances)]
        episode_rewards = [0] * instances

        envs = [self.create_env for _ in range(instances)]
        states = [env.reset() for env in envs]

        #envs = [self.create_env for _ in range(instances)]
       # states = [env.reset() for env in envs]
        for step in range(max_steps):
            for i in range(instances):
                if visualize:
                    envs[i].render()
                action = self.agent.act(states[i][0], i)
                action= max(0, min(action,  envs[i].action_space.n-1))
                if hasattr(action, "item"):
                    action = action.item()
                next_state, reward, done, _,_ = envs[i].step(action=action)
                episode_rewards[i] += reward
                episode_reward_sequences[i].append(episode_rewards[i])
                episode_step_sequences[i].append(step)
                if plot:
                    plot(episode_reward_sequences, episode_step_sequences)
                if done:
                    episode_rewards[i] = 0
                    states[i] = envs[i].reset()
                else:
                    states[i] = [next_state]

# 多进程训练子进程入口方法
def _train(create_env, instance_ids, max_steps, pipes, visualize):
    pipes = {iid: p for iid, p in zip(instance_ids, pipes)}
    actions = {iid: None for iid in instance_ids}
    create_env = cloudpickle.loads(create_env)
    envs = {iid: create_env for iid in instance_ids}
    for iid in instance_ids:
        state = envs[iid].reset()
        pipes[iid].send(RewardState(None, state))
    for step in range(max_steps):
        for iid in instance_ids:
            actions[iid] = pipes[iid].recv()
            if visualize:
                envs[iid].render()
            next_state, reward, done, _ = envs[iid].step(actions[iid])
            pipes[iid].send(RewardState(reward, None if done else next_state))
            if done:
                state = envs[iid].reset()
                pipes[iid].send(RewardState(None, state))
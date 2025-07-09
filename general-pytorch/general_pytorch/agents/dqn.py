# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:19
# @Author  : Enjoy Zhao
# @Describe ：在本文件是基于DQN算法的智能体agent，具备模型的训练、保存、预测等功能（PyTorch版本）。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from general_pytorch.core import Agent, GrEexception
from general_pytorch.policy import EpsGreedy, Greedy
from general_pytorch import memory

class DuelingMLP(nn.Module):
    def __init__(self, base_model, num_actions, dueling_type='avg'):
        super().__init__()
        # 假设base_model为nn.Sequential，最后一层为特征输出
        self.feature = nn.Sequential(*list(base_model.children())[:-1])
        last_layer = list(base_model.children())[-1]
        if hasattr(last_layer, 'in_features'):
            last_dim = last_layer.in_features
        else:
            last_dim = base_model[-1].in_features
        self.dueling_type = dueling_type
        self.value = nn.Linear(last_dim, 1)
        self.advantage = nn.Linear(last_dim, num_actions)

    def forward(self, x):
        feats = self.feature(x)
        value = self.value(feats)
        adv = self.advantage(feats)
        if self.dueling_type == 'avg':
            adv_mean = adv.mean(1, keepdim=True)
            q = value + (adv - adv_mean)
        elif self.dueling_type == 'max':
            adv_max = adv.max(1, keepdim=True)[0]
            q = value + (adv - adv_max)
        elif self.dueling_type == 'naive':
            q = value + adv
        else:
            raise GrEexception("dueling_type must be one of {'avg','max','naive'}")
        return q

class DQN(Agent):
    """
    实现一个基于dqn网络的智能体（PyTorch版本）
    """
    def __init__(self, model, actions, optimizer=None, policy=None, test_policy=None,
                 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
                 enable_double_dqn=False, enable_dueling_network=False, dueling_type='avg', device=None):
        super().__init__()
        self.actions = actions
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = EpsGreedy(0.1) if policy is None else policy
        self.test_policy = Greedy() if test_policy is None else test_policy
        self.memsize = memsize
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)
        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        # Build model
        if self.enable_dueling_network:
            self.model = DuelingMLP(model, self.actions, dueling_type=self.dueling_type)
            self.target_model = DuelingMLP(model, self.actions, dueling_type=self.dueling_type)
        else:
            self.model = model
            self.target_model = type(model)()  # assumes model is nn.Module
        self.model.to(self.device)
        self.target_model.to(self.device)
        #self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-3) if optimizer is None else optimizer
        self.loss_fn = nn.MSELoss()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def act(self, state, instance=1):
        self.model.eval()
       
        state_tensor = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
           # print(state_tensor[0].reshape(1,4))
            qvals = [self.model(state_tensor.reshape(1,4))[0].numpy()]
        if self.training:
            return self.policy.act(qvals)
        else:
            return self.test_policy.act(qvals)

    def push(self, transition, instance=0):
        self.memory.put(transition)

    def train(self, step):
        if len(self.memory) == 0:
            return
        # Update target network
        if self.target_update >= 1 and step % self.target_update == 0:
            self.target_model=self.model
        elif self.target_update < 1:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.target_update * param.data + (1.0 - self.target_update) * target_param.data)

        batch_size = min(len(self.memory), self.batch_size)
        state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)
        state_batch = torch.FloatTensor(state_batch[0][0]).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        target_qvals = np.zeros(batch_size)
        non_final_last_next_states = [es for es in end_state_batch if es is not None]
        if len(non_final_last_next_states) > 0:
            next_states = torch.FloatTensor(np.array(non_final_last_next_states)).to(self.device)
            if self.enable_double_dqn:
                q_values = self.model(next_states)
                actions = q_values.argmax(1)
                target_q_values = self.target_model(next_states)
                selected_target_q_vals = target_q_values.gather(1, actions.unsqueeze(1)).squeeze(1).cpu().detach().numpy()
            else:
                target_q_values = self.target_model(next_states)
                selected_target_q_vals = target_q_values.max(1)[0].cpu().numpy()
            non_final_mask = [s is not None for s in end_state_batch]
            target_qvals[non_final_mask] = selected_target_q_vals

        for n in reversed(range(self.nsteps)):
            rewards = np.array([b[n] for b in reward_batches])
            target_qvals *= np.array([t[n] for t in not_done_mask])
            target_qvals = rewards + (self.gamma * target_qvals)

        # Compute Q values for the taken actions
        self.model.train()
        q_pred = self.model(state_batch).reshape(batch_size,-1)
       
        q_selected = q_pred.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        target_qvals_tensor = torch.FloatTensor(target_qvals).to(self.device)

        loss = self.loss_fn(q_selected, target_qvals_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities if using prioritized replay
        if isinstance(self.memory, memory.PrioritizedExperienceReplay):
            td_error = (q_selected.detach().cpu().numpy() - target_qvals)
            traces_idxs = self.memory.last_traces_idxs()
            self.memory.update_priorities(traces_idxs, np.abs(td_error))
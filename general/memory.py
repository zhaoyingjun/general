# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 21:48
# @Author  : Enjoy Zhao
"""Abstract base class for memory"""
from collections import namedtuple,deque
import random
import numpy as np
import abc

Transition=namedtuple('Transition',['state','action','reward','next_state'])

class Memory:
    """

    """
    #@abc.abstractmethod
    def put(self,*args,**kwargs):

        raise NotImplementedError()

    #@abc.abstractmethod
    def get(self,*args,**kwargs):

        raise NotImplementedError()

    #@abc.abstractmethod
    def __len__(self):

        raise NotImplementedError()

def unpack(traces):
    """

    """
    states=[t[0].state for t in traces]

    actions=[t[0].action for t in traces]

    rewards=[[e.reward for e in t] for t in traces]

    end_states=[t[-1].next_state for t in traces]

    not_done_mask=[[1 if n.next_state is not None else 0 for n in t]for t in traces]

    return states,actions,rewards,end_states,not_done_mask

class OnPolicy(Memory):
    """

    """
    def __init__(self,steps=1,instances=1):
        self.bufffers=[[] for _ in range(instances)]
        self.steps=steps
        self.instances=instances
    def put(self,transition,instance=0):
        """
        """
        self.bufffers[instance].append(transition)

    def get(self):
        """

        """
        traces=[list(tb) for tb in self.bufffers]

        self.bufffers=[[] for _ in range(self.instances)]

        return unpack(traces)

    def __len__(self):
        """
        """

        return sum([len(b)-self.steps+1 for b in self.bufffers])

class ExperienceReplay(Memory):
    """

    """
    def __init__(self,capacity,steps=1,exclude_boundaries=False):
        self.traces=deque(maxlen=capacity)
        self.buffer=[]
        self.steps=steps
        self.exclude_boundaries=exclude_boundaries

    def put(self,transition):


        self.buffer.append(transition)

        if len(self.buffer)<self.steps:return

        self.traces.append(tuple(self.buffer))


        if self.exclude_boundaries and transition.next_state is None:
            self.buffer=[]
            return
        self.buffer=self.buffer[1:]

    def get(self,batch_size):

        traces=random.sample(self.traces,batch_size)
        return unpack(traces)

    def __len__(self):

        return len(self.traces)



EPS=1e-3

class PrioritizedExperienceReplay(Memory):

    def __init__(self,capacity,steps=1,exclude_boundaries=False,prob_alpha=0.6):




        self.traces=[]

        self.priorities=np.array([])

        self.buffer=[]

        self.capacity=capacity

        self.steps=steps
        self.exclude_boundaries=exclude_boundaries

        self.prob_alpha=prob_alpha

        self.traces_idxs=[]

    def put(self,transition):

        self.buffer.append(transition)

        if len(self.buffer)<self.steps:return

        if len(self.traces)<self.capacity:

            self.traces.append(tuple(self.buffer))
            self.priorities=np.append(self.priorities,EPS if self.priorities.size==0 else self.priorities.max())
        else:

            idx=np.argmin(self.priorities)

            self.traces[idx]=tuple(self.buffer)
            self.priorities[idx]=self.priorities.max()

        if self.exclude_boundaries and transition.next_state is None:

            self.buffer=[]

            return
        self.buffer=self.buffer[1:]

    def get(self,batch_size):


        probs=self.priorities**self.prob_alpha

        probs /=probs.sum()

        self.traces_idxs=np.random.choice(len(self.traces),batch_size,p=probs,replace=False)

        traces=[self.traces[idx] for idx in self.traces_idxs]

        return unpack(traces)

    def last_traces_idxs(self):

        return self.traces_idxs.copy()

    def update_priorities(self,trace_idxs,new_priorities):

        self.priorities[trace_idxs]=new_priorities+EPS

    def __len__(self):

        return len(self.traces)














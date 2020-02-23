# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 21:48
# @Author  : Enjoy Zhao
# @Describe ：在本文件中主要存储器基类和不同的记忆回放算法的存储器，存储器的作用是存储在强化学习中的当前状态、执行动作、环境反馈、下一个状态，用于智能体预测网络的训练。
# 目前已经实现OnPolicy、ExperienceReplay、PrioritizedExperienceReplay

from collections import namedtuple,deque
import random
import numpy as np
import abc

""" 定义所存储的记录格式，以元组的方式存储当前状态、执行动作、环境反馈、下一个状态 """
Transition=namedtuple('Transition',['state','action','reward','next_state'])
"""实现一个Memory基类，并要求后续的继承该基类的方法类必须包括put、get、 __len__方法。"""
class Memory:

    @abc.abstractmethod
    def put(self,*args,**kwargs):

        raise NotImplementedError()

    @abc.abstractmethod
    def get(self,*args,**kwargs):

        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):

        raise NotImplementedError()
"""定义一个unpack方法，作用是把从存储器取出的记录给拆解，然后返回。"""
def unpack(traces):
   #拆解出记录中的states 
    states=[t[0].state for t in traces]
   # 拆解出记录中的actions
    actions=[t[0].action for t in traces]
   # 拆解出记录中的rewards
    rewards=[[e.reward for e in t] for t in traces]
   #拆解出记录中的end_states
    end_states=[t[-1].next_state for t in traces]
   #根据下一个状态是否存在来判断是否记录终止了，返回相应的标志位
    not_done_mask=[[1 if n.next_state is not None else 0 for n in t]for t in traces]
   #返回拆解出的值
    return states,actions,rewards,end_states,not_done_mask

"""实现一个原始的存储器方法类，实现记录的存储put、获取get、以及存储的记录数量，存储和读取都是全存全取的，没有优化算法。"""
class OnPolicy(Memory):
    """
    初始化参数，包括steps和instances,steps是指存储记录的步数，instances是指实例的数量，也就是智能体的数量。
    """
    def __init__(self,steps=1,instances=1):
        self.bufffers=[[] for _ in range(instances)]
        self.steps=steps
        self.instances=instances
    def put(self,transition,instance=0):
        """
        因为初始化instances=1,也就是说只有一个智能体，那么我们就将记录全部存储在bufffers[0]中，transition就是记录数据
        """
        self.bufffers[instance].append(transition)

    def get(self):
        """
        将所有的记录从bufffer中取出，清空存储器，然后对记录进行拆解返回。
        """
        traces=[list(tb) for tb in self.bufffers]

        self.bufffers=[[] for _ in range(self.instances)]

        return unpack(traces)

    def __len__(self):
        """
        统计存储器中的记录数量并返回
        """

        return sum([len(b)-self.steps+1 for b in self.bufffers])

"""实现一个带有经验回放的存储器方法类，同样实现记录的存储put、获取get、以及存储的记录数量的统计，不过在存储记录和获取记录时并非全存全取的，使用了算法以打破记录的连续性和关联性，
   因为如果不这样做，算法在连续一段时间内基本朝着同一个方向做gradient descent，那么同样的步长下这样直接计算gradient就有可能不收敛，因此这样做可以有助于模型收敛。同时打破连续性和关联性还可以增加网络的泛化性。
"""
class ExperienceReplay(Memory):
    """
    初始化参数，capacity是存储器的存储容量,steps存储记录的步数,exclude_boundaries是标记是否存储边界值,traces是存储队列
    """
    def __init__(self,capacity,steps=1,exclude_boundaries=False):
        self.traces=deque(maxlen=capacity)
        self.buffer=[]
        self.steps=steps
        self.exclude_boundaries=exclude_boundaries

    def put(self,transition):

        self.buffer.append(transition)
        #达到一定的记录步数之后，将存储buffer存入到存储队列trances中去
        if len(self.buffer)<self.steps:return
        self.traces.append(tuple(self.buffer))

        #如果不包含边界数据且记录中的下一个状态不存在，则将buffer清空
        if self.exclude_boundaries and transition.next_state is None:
            self.buffer=[]
            return
        self.buffer=self.buffer[1:]

    def get(self,batch_size):
    #按照batch_size的大小随机从存储器中取出记录值并进行拆解后返回
        traces=random.sample(self.traces,batch_size)
        return unpack(traces)

    def __len__(self):
    #返回存储队列中的存储记录数量
        return len(self.traces)

""" 
经验回放的存储器按照全部完全随机的算法将记录的连续性和关联性打破，虽然可以提高模型收敛和泛化性能，但是由于是完全随机带来了完全不确定性会导致特征稀疏的问题，因此需要增加一个优先策略，来增加有限的连续性。 
"""
#EPS 是一个优先级默认值
EPS=1e-3

class PrioritizedExperienceReplay(Memory):

    """
    初始化参数，包括capacity是存储器的存储容量,steps存储记录的步数,exclude_boundaries是标记是否存储边界值,prob_alpha 概率系数,traces存储数组以及trances_indexs索引数组,priorities优先级数组
    """

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

        # 达到一定的记录步数之后，且没有达到最大存储规模，将存储buffer存入到trances中去，并新增优先级到priorities中去。

        if len(self.buffer)<self.steps:return

        if len(self.traces)<self.capacity:

            self.traces.append(tuple(self.buffer))
            #如果优先级数组内为空，则使用默认值EPS，否则取priorities的最大值
            self.priorities=np.append(self.priorities,EPS if self.priorities.size==0 else self.priorities.max())
        else:
            #如果存储器达到最大存储规模上线了，则使用当前记录替换掉存储器中优先级最小的记录,并将原本记录的优先级更新成最高优先级。
            idx=np.argmin(self.priorities)

            self.traces[idx]=tuple(self.buffer)

            self.priorities[idx]=self.priorities.max()

        # 如果不包含边界数据且记录中的下一个状态不存在，则将buffer清空

        if self.exclude_boundaries and transition.next_state is None:

            self.buffer=[]

            return
        self.buffer=self.buffer[1:]

    def get(self,batch_size):

        #将优先级数组进行归一化，将优先级转换为概率分布，然后使用随机选择函数依照优先级概率分布确定取出记录的索引，最后依据索引从存储器中取出记录并拆解后返回。
        probs=self.priorities**self.prob_alpha

        probs /=probs.sum()

        self.traces_idxs=np.random.choice(len(self.traces),batch_size,p=probs,replace=False)

        traces=[self.traces[idx] for idx in self.traces_idxs]

        return unpack(traces)

    def last_traces_idxs(self):
        return self.traces_idxs.copy()
    #定义优先级更新函数，为了防止优先级为0，因此需要加上EPS
    def update_priorities(self,trace_idxs,new_priorities):

        self.priorities[trace_idxs]=new_priorities+EPS

    def __len__(self):
        # 返回存储中的记录数量
        return len(self.traces)














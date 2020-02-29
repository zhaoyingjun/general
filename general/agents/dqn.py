# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:19
# @Author  : Enjoy Zhao
# @Describe ：在本文件是基于DQN算法的智能体agent，具备模型的训练、保存、预测等功能。

import tensorflow as tf
import numpy as np

from general.core import Agent,GrEexception
from general.policy import EpsGreedy,Greedy
from general import memory

class DQN(Agent):

    """
    实现一个基于dqn网络的智能体
    """
    def __init__(self,model, actions, optimizer=None, policy=None, test_policy=None,
				 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
				 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):

        """

        :param model:是我们定义的智能体的神经网络
        :param actions:是需要执行的动作的空间，其实就是智能体输出的动作集中需要包含动作的数量
        :param optimizer:优化器
        :param policy:是配置训练时action的选择策略
        :param test_policy:配置测试时action的选择策略
        :param memsize:记忆存储器的大小
        :param target_update:配置eval网络训练多少次之后更新一次target网络
        :param gamma:是一个系数
        :param batch_size:批量训练每批次的大小
        :param nsteps:可以理解为向前看的深度，比如nsteps=2，表示模型取的前两个action步骤的reward
        :param enable_double_dqn:配置是否使用double_dqn
        :param enable_dueling_network:配置是否使用dueling网络
        :param dueling_type:配置dueling网络的策略
        """
        #以下是进行参数的初始化，首先是对动作种类的数量进行初始化
        self.actions=actions
        #对神经网络的优化器进行初始化，默认使用adam
        self.optimizer = tf.keras.optimizers.Adam(lr=3e-3) if optimizer is None else optimizer
        #初始化训练状态下的action选择策略，默认是随机贪婪算法。
        self.policy = EpsGreedy(0.1) if policy is None else policy
        #初始化测试状态下的action选择策略，默认是原始贪婪算法。
        self.test_policy = Greedy() if test_policy is None else test_policy
        #初始化记忆器的存储空间大小
        self.memsize = memsize
        #初始化记忆回放策略，采用优先级经验回放
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        #初始化输出向量
        raw_output=model.layers[-1].output
        #如果配置使用dueling网络，则在输出层上一分为二，分为两个中间输出层网络，分别是价值网络和优势网络，然后将两个中间输出层的结果线性组合输出。
        #所谓价值网络就是输出结果与状态S有关，与执行动作action无关，而优势网络与状态和执行动作都有关系，其实就是朴素dqn的输出。
        if self.enable_dueling_network:
            #使用Dnese构建中间输出层，与dqn相比其输出维度加1，其中输出的结果a[:,0]是价值网络的输出，a[:,1:]是优势网络的输出。
            dueling_layer=tf.keras.layers.Dense(self.actions+1,activation='liner')(raw_output)
            #如果dueling策略选择均值策略，则在进行两个网络输出线性组合前，对优势网络按照均值策略进行优化处理。
            if self.dueling_type == 'avg':
                f = lambda a: tf.expand_dims(a[:, 0], -1) + (a[:, 1:] - tf.reduce_mean(a[:, 1:],axis=1,keepdims=True))
            elif self.dueling_type == 'max':
                # 如果dueling策略选择最大策略，则在进行两个网络输出线性组合前，对优势网络按照最大值策略进行优化处理。
                f = lambda a: tf.expand_dims(a[:, 0], -1) + (a[:, 1:] - tf.reduce_max(a[:, 1:],axis=1,keepdims=True))
            elif self.dueling_type == 'naive':
                # 如果dueling策略选择原生策略，则在进行两个网络输出线性组合前，不对优势网络进行任何处理。
                f = lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:]
            else:
                raise GrEexception("dueling_type must be one of {'avg','max','naive'}")

            output_layer=tf.keras.layers.Lambda(f,output_shape=(self.actions,))(dueling_layer)


        else:
            #如果不是dueling网络，则按照dqn的标准网络输出。
            output_layer=tf.keras.layers.Dense(self.actions,activation='linear')(raw_output)
        #使用tf.keras.Model构建神经网络模型
        self.model=tf.keras.Model(inputs=model.input,outputs=output_layer)

        #定义一个Loss函数，用于计算模型预测输出结果的loss
        def masked_q_loss(data,y_pred):

            """
            :param data:其中包含action和其对应reward数据
            :param y_pred:预测的action数据

            """
            #从data中取出action和reward
            action_batch,target_qvals=data[:,0],data[:,1]

            #接下来需要构建一个特殊数组action_idxs，能够根据action_idxs gather出在y_pred中与action_idx对应的q值。
            #首先根据action_batch的长度构建一个递增数组，比如[0，1，2，3]
            seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
            #然后通过将seq与action_batch(批量取出的动作）进行堆叠组合多维数组，对多维数组进行转置，这样处理是为了能够与y_pred进行gather_nd运算
            action_idxs=tf.transpose(tf.stack([seq,tf.cast(action_batch,tf.int32)]))
            #最后通过gather_nd运算得到我们需要的qvals，也就是预测的动作的q值。
            qvals=tf.gather_nd(y_pred,action_idxs)
            #这里是判断一下是否使用PrioritizedExperienceReplay策略进行记忆回放，如果是则对记忆回放按照如下处理
            if isinstance(self.memory, memory.PrioritizedExperienceReplay):
                #定义一个优先级更新函数，更新记录的优先级
                def update_priorities(_qvals,_target_qvals,_traces_idxs):

                    """
                    这里需要更新优先级的计算是计算target_qvals和预测得到qvals的差的绝对值，差值越大优先级越高，这也是引导在记忆回放时能够
                    覆盖到陌生领域。以下是算法的实现过程。

                    """
                    #计算得到优先级的值
                    td_error=tf.abs((_target_qvals-_qvals).numpy())
                    #得到记录对应的idx
                    _traces_idxs=(tf.cast(_traces_idxs,tf.int32)).numpy()
                    #调用memory中PrioritizedExperienceReplay的优先度更新方法进行更新
                    self.memory.update_priorities(_traces_idxs,td_error)

                    return _qvals
                #使用tf.py_function调用update_priorities更新所需要更新的优先度
                qvals=tf.py_function(func=update_priorities,inp=[qvals,target_qvals,data[:,2]],Tout=tf.float32)
             #使用mes作为loss计算损失值
            return tf.keras.losses.mse(qvals,target_qvals)
        #最后编译网络模型。
        self.model.compile(optimizer=self.optimizer,loss=masked_q_loss)

        # q_eval 网络和target网络是完全一样的网络结构，因此我们直接复制q_eval即可
        #复制网络模型
        self.target_model=tf.keras.models.clone_model(self.model)
        #配置网络参数
        self.target_model.set_weights(self.model.get_weights())

    def save(self,filename,overwrite=False,save_format='h5'):

        """
        save方法是用于保存网络模型到特定文件

        """
        self.model.save_weights(filename,overwrite=overwrite,save_format=save_format)


    def act(self,state,instance=0):

        """
        act 方法是根据网络模型和输入状态数据，预测最有的执行动作

        """
        qvals=self.model.predict(np.array([state]))[0]
        #使用动作选择算法对输出的结果选出最终的动作
        return self.policy.act(qvals) if self.training else self.test_policy.act(qvals)


    def push(self,transition,instance=0):

        """
        push 方法是将记录存入到记忆存储器中。
        """
        self.memory.put(transition)

    def train(self,step):

        """
         train方法是实现了对智能体神经网络的训练过程

        """
        #首先判断记忆存储器中的记录是否存在，如果存在则继续进行训练。
        if len(self.memory)==0:
            return
        #判断target网络是否需要更新，如果达到更新条件则进行更新。
        if self.target_update>=1 and step % self.target_update==0:

            self.target_model.set_weights(self.model.get_weights())
       #如果target_update小于1，则直接将target_update作为系数，组合target_model和model的参数进行更新。
        elif self.target_update<1:

            mw=np.array(self.model.get_weights())

            tmw=np.array(self.target_model.get_weights())

            self.target_model.set_weights(self.target_update*mw+(1-self.target_update)*tmw)

        #将batch_size取最小范围，以保证在记忆存储器的记录数量不小于batch_size
        batch_size=min(len(self.memory),self.batch_size)
        #从记忆存储器中批量的取数据，大小是batch_size
        state_batch,action_batch,reward_batches,end_state_batch,not_done_mask=self.memory.get(batch_size)
        #使用全零数组初始化target_qvals
        target_qvals=np.zeros(batch_size)
        #取出非终止状态的下一个状态
        non_final_last_next_states=[es for es in end_state_batch if es is not None]

        if len(non_final_last_next_states)>0:
           #如果设置的是double_dqn，则进行如下的过程计算selected_target_q_vals
            if self.enable_double_dqn:
               #首先使用q_eval网络将non_final_last_next_states中的状态作为输入，进而预测q_values
                q_values=self.model.predict_on_batch(np.array(non_final_last_next_states))
               #使用argmax获得q_values最大值对应的actions。
                actions=tf.cast(tf.argmax(q_values,axis=1),tf.int32)
                #使用target网络将non_final_last_next_states中的状态作为输入，进而预测target_q_values
                target_q_values=self.target_model.predict_on_batch(np.array(non_final_last_next_states))
                 #接下来也是一个gather_nd的过程，现实组合一个特殊的数组，目的是找到在target_q_values中actions对应的值
                selected_target_q_vals=tf.gather_nd(target_q_values,tf.transpose(tf.stack([tf.range(len(target_q_values)),actions])))

            else:
                #如果不是使用double_dqn，则直接使用target网络将non_final_last_next_states中的状态作为输入，进而预测target_q_values,并取最大值。

                selected_target_q_vals=self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)

            #将end_state_batch为非None的数值取出
            non_final_mask=list(map(lambda s:s is not None,end_state_batch))
            #将selected_target_q_vals值更新到状态非None对应的target_qvals中
            target_qvals[non_final_mask]=selected_target_q_vals

        #下面是根据nsteps的配置，取将前n个状态的reward引入到target_qvals中
        for n in reversed(range(self.nsteps)):
            #取reward_batches中的前n个reward
            rewards=np.array([b[n] for b in reward_batches])
            #将target_qvals与not_done_mask中前n个值组成的数组乘机得到target_qvals
            target_qvals*=np.array([t[n] for t in not_done_mask])
            #将前面两步计算得到的rewards和target_qvals与一个系数的乘积相加得到最终的target_qvals
            target_qvals=rewards+(self.gamma*target_qvals)
        #将action_batch、target_qvals组合成lossdata，计算loss
        loss_data=[action_batch,target_qvals]

        #如果是使用优先级记忆回放策略loss_data还需要加上记忆存储器的最后记录的indexs
        if isinstance(self.memory, memory.PrioritizedExperienceReplay):

            loss_data.append(self.memory.last_traces_idxs())
        #将数据灌入模型，进行训练。
        self.model.train_on_batch(np.array(state_batch),tf.transpose(tf.stack(loss_data)))







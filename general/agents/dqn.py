# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 22:19
# @Author  : Enjoy Zhao


import tensorflow as tf
import numpy as np

from general.core import Agent,GrEexception
from general.policy import EpsGreedy,Greedy
from general import memory

class DQN(Agent):

    """

    """
    def __init__(self,model, actions, optimizer=None, policy=None, test_policy=None,
				 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
				 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):

        """

        :param model:
        :param actions:
        :param optimizer:
        :param policy:
        :param test_policy:
        :param memsize:
        :param target_update:
        :param gamma:
        :param batch_size:
        :param nsteps:
        :param enable_double_dqn:
        :param enable_dueling_network:
        :param dueling_type:
        """

        self.actions=actions

        self.optimizer = tf.keras.optimizers.Adam(lr=3e-3) if optimizer is None else optimizer

        self.policy = EpsGreedy(0.1) if policy is None else policy
        self.test_policy = Greedy() if test_policy is None else test_policy

        self.memsize = memsize
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        # Extension options
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        raw_output=model.layers[-1].output

        if self.enable_dueling_network:

            dueling_layer=tf.keras.layers.Dense(self.actions+1,activation='liner')(raw_output)
            if self.dueling_type == 'avg':
                f = lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.reduce_mean(a[:, 1:],
                                                                                      axis=1,
                                                                                      keepdims=True)
            elif self.dueling_type == 'max':
                f = lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.reduce_max(a[:, 1:],
                                                                                     axis=1,
                                                                                     keepdims=True)
            elif self.dueling_type == 'naive':
                f = lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:]
            else:
                raise GrEexception("dueling_type must be one of {'avg','max','naive'}")

            output_layer=tf.keras.layers.Lambda(f,output_shape=(self.actions,))(dueling_layer)

        else:
            output_layer=tf.keras.layers.Dense(self.actions,activation='linear')(raw_output)

        self.model=tf.keras.Model(inputs=model.input,outputs=output_layer)


        def masked_q_loss(data,y_pred):

            """


            :param data:
            :param y_pred:
            :return:
            """

            action_batch,target_qvals=data[:,0],data[:,1]


            seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)

            action_idxs=tf.transpose(tf.stack([seq,tf.cast(action_batch,tf.int32)]))

            qvals=tf.gather_nd(y_pred,action_idxs)

            if isinstance(self.memory, memory.PrioritizedExperienceReplay):

                def update_priorities(_qvals,_target_qvals,_traces_idxs):

                    """


                    """

                    td_error=np.abs((_target_qvals-_qvals).numpy())

                    _traces_idxs=(tf.cast(_traces_idxs,tf.int32)).numpy()

                    self.memory.update_priorities(_traces_idxs,td_error)

                    return _qvals
                qvals=tf.py_function(func=update_priorities,inp=[qvals,target_qvals,data[:,2]],Tout=tf.float32)

            return tf.keras.losses.mse(qvals,target_qvals)

        self.model.compile(optimizer=self.optimizer,loss=masked_q_loss)

        # Clone model to use for delayed Q targets

        self.target_model=tf.keras.models.clone_model(self.model)

        self.target_model.set_weights(self.model.get_weights())

    def save(self,filename,overwrite=False):

        """

        :param filename:
        :param overwrite:
        :return:
        """
        self.model.save_weights(filename,overwrite=overwrite)


    def act(self,state,instance=0):

        """

        :param state:
        :param instance:
        :return:
        """
        qvals=self.model.predict(np.array([state]))[0]

        return self.policy.act(qvals) if self.training else self.test_policy.act(qvals)


    def push(self,transition,instance=0):

        """

        :param transition:
        :param instance:
        :return:
        """
        self.memory.put(transition)

    def train(self,step):

        """

        :param step:
        :return:
        """
        if len(self.memory)==0:
            return

        if self.target_update>=1 and step % self.target_update==0:

            self.target_model.set_weights(self.model.get_weights())

        elif self.target_update<1:

            mw=np.array(self.model.get_weights())

            tmw=np.array(self.target_model.get_weights())

            self.target_model.set_weights(self.target_update*mw+(1-self.target_update)*tmw)


        batch_size=min(len(self.memory),self.batch_size)

        state_batch,action_batch,reward_batches,end_state_batch,not_done_mask=self.memory.get(batch_size)

        target_qvals=np.zeros(batch_size)

        non_final_last_next_states=[es for es in end_state_batch if es is not None]

        if len(non_final_last_next_states)>0:

            if self.enable_double_dqn:

                q_values=self.model.predict_on_batch(np.array(non_final_last_next_states))
                actions=np.argmax(q_values,axis=1)

                target_q_values=self.target_model.predict_on_batch(np.array(non_final_last_next_states))

                selected_target_q_vals=tf.gather_nd(target_q_values,tf.transpose(tf.stack([tf.range(len(target_q_values)),actions])))

            else:

                selected_target_q_vals=self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)


            non_final_mask=list(map(lambda s:s is not None,end_state_batch))

            target_qvals[non_final_mask]=selected_target_q_vals


        for n in reversed(range(self.nsteps)):

            rewards=np.array([b[n] for b in reward_batches])

            target_qvals*=np.array([t[n] for t in not_done_mask])
            target_qvals=rewards+(self.gamma*target_qvals)

        loss_data=[action_batch,target_qvals]


        if isinstance(self.memory, memory.PrioritizedExperienceReplay):

            loss_data.append(self.memory.last_traces_idxs())

        self.model.train_on_batch(np.array(state_batch),np.stack(loss_data).transpose())







# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 00:43
# @Author  : Enjoy Zhao
from urllib3 import PoolManager
import json

class Proxy(object):

   def __init__(self,url,observation_space=None,init_state=None):

       self.url=url

       self.observation_space=observation_space

       self.init_state=init_state
   def step(self,action):
       """
       return
       """
       http=PoolManager(num_pools=1,headers=None)
       data = {'action':str(action)}

       response_data=json.loads(http.request('PUT',self.url,data).data)

       reward=json.loads(response_data).get('reward')
       next_state = json.loads(response_data).get('next_state')
       done=json.loads(response_data).get('done')

       return next_state,reward,done,None


   def render(self):

       #render在gym的游戏中是游戏图形化展示的方法，在使用url时该方法不适用

      return

   def reset(self):

       return self.init_state

# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 00:43
# @Author  : Enjoy Zhao
from urllib3 import PoolManager
import json

class Proxy:

   def __init__(self,url,GET=True,observation_spac=None,init_state=None):

       self.url=url

       self.GET=GET

       self.observation_spac=observation_spac

       self.init_state=init_state
   def step(self,action):
       """
       return
       """
       http=PoolManager(num_pools=100,headers=None)
       data = json.dumps({'action':action})
       if self.GET :
         response=http.request('GET',self.url,data)
       else:
         response = http.request('POST', self.url, data)

       reward=response.data.decode()['reward']
       state=response.data.decode()['state']

       return reward,state


   def observation_space(self):


       return self.observation_spac


   def render(self):


       return

   def reset(self):

       return self.init_state

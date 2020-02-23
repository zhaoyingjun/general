# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 00:43
# @Author  : Enjoy Zhao
#@Describe：本文件是定义了一个代理器，其作用是调用外部环境的url，进行与外部环境的交互。
from urllib3 import PoolManager
import json
import numpy as np

class Proxy(object):
#定义初始化函数，将url init_state进行初始化
   def __init__(self,url,init_state):

       self.url=url
       self.observation_space=np.array(init_state)
       self.init_state=init_state
#定义step方法，将action传送给外部环境并获得外部环境的反馈
   def step(self,action):
       #使用urllib3的PoolManager来构建一个http池
       http=PoolManager(num_pools=1,headers=None)

       #将action构造乘json,通过PUT方法将数据传送给外部环境服务接口
       data = {'action':str(action)}
       response_data=json.loads(http.request('PUT',self.url,data).data)
       #解析外部环境服务返回的数值，分别是reward、next_state、done
       reward=json.loads(response_data).get('reward')
       next_state = json.loads(response_data).get('next_state')
       done=json.loads(response_data).get('done')

       return next_state,reward,done,None


   def render(self):
       #render在gym的游戏中是游戏图形化展示的方法，在使用url时该方法不适用
      return
   #定义初始化函数，用于获得外部环境的初始化状态
   def reset(self):
       return self.init_state

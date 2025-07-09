# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 15:28
# @Author  : Enjoy Zhao
# @Describe ：本文件是基于flask和gym，模拟一个外部环境服务，用于展示如何使用proxy通过url与外部的环境进行交互和训练
from flask import Flask,request
from flask import jsonify
import numpy as np
from flask_restful import Api, Resource
import json
import gym
#创建flask项目
app=Flask('urldemo')
api=Api(app)
#使用gym创建一个游戏，用于模拟外部环境
create_env=lambda: gym.make('CartPole-v0').unwrapped
dym_env=create_env()
dym_env.reset()


#定义一个json encoder类，用于处理json数据
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

s=[0]
#定义env方法类，实现Put方法，将action输入gym环境，并将gym环境返回的数据进行封装后返回。
class env(Resource):
  def put(self):
    #从request中获取action
    action=request.form['action']
    action=np.int64(action)
    #将action输入到外部环境中
   # dym_env.render()
    next_state,reward,done,_=dym_env.step(action=action)
    if done:
        dym_env.reset()
    #将返回的数据封装成json数据后返回
    data=json.dumps({
        'reward':reward,
        'next_state':next_state,
        'done':done

    },cls=NpEncoder)


    return jsonify(data)
#设置api路由
api.add_resource(env,'/')

if __name__=='__main__':
    app.run(debug=True)

# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 15:28
# @Author  : Enjoy Zhao
from flask import Flask,request
from flask import jsonify
import numpy as np
from flask_restful import Api, Resource
import json
import gym
app=Flask('test')
api=Api(app)
create_env=lambda: gym.make('CartPole-v0').unwrapped
dym_env=create_env()
dym_env.reset()


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
class env(Resource):
  def put(self):
    action=request.form['action']
    action=np.int64(action)
    next_state,reward,done,_=dym_env.step(action=action)
    if done:
        dym_env.reset()

    data=json.dumps({
        'reward':reward,
        'next_state':next_state,
        'done':done

    },cls=NpEncoder)
    print(data)

    return jsonify(data)
api.add_resource(env,'/')

if __name__=='__main__':
    app.run(debug=True)

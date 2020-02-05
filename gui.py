# coding = utf-8
import wx
import time
import threading
from client import client
import gym
import matplotlib
matplotlib.use("macOSX")
from multiprocessing import Process, Queue
dummy_env = lambda: gym.make('CartPole-v0').unwrapped
dummy_env=dummy_env()


# s = client()
data_x={'dense_num':3,
        'cell_num':16,
        'activation':'relu',
        'train_steps':3000,
        'model_name':'model_name',
        'algorithm_type':'dqn'

}




class RewardThread(threading.Thread):

    def __init__(self, parent,env):
        """
        :param parent:  主线程UI
        :param timer:  计时器
        """
        super(RewardThread, self).__init__()
        self.parent = parent
        self.env=env
        self.setDaemon(True)  # 设置为守护线程， 即子线程是守护进程，主线程结束子线程也随之结束。

    def run(self):
        count = 0
        while count < data_x['train_steps']:
            count = count + 1
            time.sleep(5)
            reward=self.env.ep_reward
            step=self.env.ep_step
            self.parent.reward_data=reward
            self.parent.step_data=step
            wx.CallAfter(self.parent.update_train_log)

class TrainThread(threading.Thread):

    def __init__(self, parent,client):
        super(TrainThread, self).__init__()
        self.parent = parent
        self.client=client
        self.setDaemon(True)
    def run(self):
       # s=self.parent.create_env()
        self.client.train()


class AppUI(wx.Panel):
        def __init__(self, parent):
            wx.Panel.__init__(self, parent)

            # 创建一些sizer
            self.client = None
            self.dummy_env=None
            self.step_data=0
            self.reward_data = 0

            mainSizer = wx.BoxSizer(wx.VERTICAL)
            grid = wx.GridBagSizer(vgap=7, hgap=7)  # 虚拟网格sizer，此处指定了行和列之间的间隙
            hSizer = wx.BoxSizer(wx.HORIZONTAL)

            self.quote = wx.StaticText(self, label="", pos=(20, 20))
            grid.Add(self.quote, pos=(0, 0))

            # 展示训练过程的rewards
            self.logger = wx.TextCtrl(self, pos=(300, 20), size=(200, 350),
                                      style=wx.TE_MULTILINE | wx.TE_READONLY)



            # 编辑组件
            self.lblname1 = wx.StaticText(self, label="项目名称：", pos=(20, 60))
            grid.Add(self.lblname1, pos=(1, 0))
            self.editname1 = wx.TextCtrl(self, value="请输入项目名称", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname1, pos=(1, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText1, self.editname1)


            self.lblname2 = wx.StaticText(self, label="深度神经网络层数", pos=(30, 60))
            grid.Add(self.lblname2, pos=(2, 0))
            self.editname2 = wx.TextCtrl(self, value="请输入正整数", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname2, pos=(2, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText2, self.editname2)


            self.lblname3 = wx.StaticText(self, label="深度神经网络神经元数量", pos=(30, 60))
            grid.Add(self.lblname3, pos=(3, 0))
            self.editname3 = wx.TextCtrl(self, value="请输入正整数", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname3, pos=(3, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText3, self.editname3)


            self.lblname4 = wx.StaticText(self, label="模型训练步数", pos=(30, 60))
            grid.Add(self.lblname4, pos=(4, 0))
            self.editname4 = wx.TextCtrl(self, value="请输入正整数", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname4, pos=(4, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText4, self.editname4)

            # 组合框组件
            self.sampleList = ['relu', 'sigmod','softmax']
            self.lblhear = wx.StaticText(self, label="神经网络激活函数", pos=(20, 70))
            grid.Add(self.lblhear, pos=(5, 0))
            self.edithear = wx.ComboBox(self, pos=(150, 70), size=(95, -1), choices=self.sampleList,
                                        style=wx.CB_DROPDOWN)
            grid.Add(self.edithear, pos=(5, 1))
            self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox1, self.edithear)


            # 组合框组件
            self.sampleList = ['dqn', 'ddpg','ppo']
            self.lblhear = wx.StaticText(self, label="选择强化学习算法", pos=(20, 70))
            grid.Add(self.lblhear, pos=(6, 0))
            self.edithear = wx.ComboBox(self, pos=(150, 70), size=(95, -1), choices=self.sampleList,
                                        style=wx.CB_DROPDOWN)
            grid.Add(self.edithear, pos=(6, 1))
            self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox2, self.edithear)


            #


            #
            self.insure = wx.StaticText(self, label="请选择Gym游戏", pos=(20, 180))
            grid.Add(self.insure, pos=(7, 0), span=(1, 2))


            radioList = ['CartPole', 'MountCar']
            rb = wx.RadioBox(self, label="", pos=(20, 210),
                             choices=radioList, majorDimension=3,
                             style=wx.RA_SPECIFY_COLS)
            grid.Add(rb, pos=(8, 0), span=(1, 2))
            self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, rb)
            # 运行按钮
            self.button_1 = wx.Button(self, label="训练", pos=(200, 325))
            self.Bind(wx.EVT_BUTTON, self.on_click_start, self.button_1)

            self.button_2 = wx.Button(self, label="测试", pos=(200, 350))
            self.Bind(wx.EVT_BUTTON, self.on_click_start, self.button_2)

            hSizer.Add(grid, 0, wx.ALL, 5)
            hSizer.Add(self.logger)
            mainSizer.Add(hSizer, 0, wx.ALL, 5)
            mainSizer.Add(self.button_1, 0, wx.CENTER)
            mainSizer.Add(self.button_2, 1, wx.CENTER)
            self.SetSizerAndFit(mainSizer)

        def update_train_log(self):
            self.logger.AppendText('----------------------\n')
            self.logger.AppendText('|Train_steps  | %d\n' % self.step_data)
            self.logger.AppendText('|Train_reward| %d\n' % self.reward_data)

        def EvtRadioBox(self, event):

            if event.GetInt()==0:

                self.dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = self.dummy_env()
            elif event.GetInt()==1:
                dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = dummy_env()



        def EvtComboBox1(self, event):
            if event.GetString()=='dqn':
                data_x['algorithm_type']='dqn'
            elif event.GetString()=='ddpg':
                data_x['algorithm_type'] = 'ddpg'

            elif event.GetString()=='ppo':
                data_x['algorithm_type'] = 'ppo'


        def EvtComboBox2(self, event):
            if event.GetString()=='relu':
                data_x['activation']='dqn'
            elif event.GetString()=='sigmod':
                data_x['activation'] = 'sigmod'

            elif event.GetString()=='softmax':
                data_x['activation'] = 'softmax'




        #

        def EvtText1(self, event):
            data_x['model_name']=event.GetString()

        def EvtText2(self, event):
            data_x['dense_num']=int(event.GetString())

        def EvtText3(self, event):
            data_x['cell_num'] = int(event.GetString())

        def EvtText4(self, event):
            data_x['train_steps'] = int(event.GetString())
            print(data_x['train_steps'])

        def EvtText5(self, event):
            data_x['activation'] = event.GetString()





        def on_click_start(self,event):
            self.env = client(dense_num=data_x['dense_num'], cell_num=data_x['cell_num'], activation=data_x['activation'], train_steps=data_x['train_steps'],
                       dummy_env=self.dummy_env, model_name=data_x['model_name'], algorithm_type=data_x['algorithm_type'])


            self.progress_1 = RewardThread(self,self.env)
            self.progress_1.start()
            self.train = TrainThread(self,self.env)
            self.train.start()

if __name__ == "__main__":
        # app = wx.App(False)
        # AppUI(None).Show()
        # app.MainLoop()
        app = wx.App(False)
        frame = wx.Frame(None, title="General项目配置页面", size=(600, 400))
        nb = wx.Notebook(frame)

        nb.AddPage(AppUI(nb), "gym环境配置界面")
        frame.Show()
        app.MainLoop()



# coding = utf-8
# @Time    : 2020-01-31 22:18
# @Author  : Enjoy Zhao
# @Describe ：本文件使用wxpython编写来一个gui，用于配置general项目参数和展示训练过程
import wx
import time
import threading
from client import client
import gym
import matplotlib
from proxy import Proxy
import configparser
matplotlib.use("macOSX")
import os

wildcard = u"Python 文件 (*.py)|*.py|" \
           u"编译的 Python 文件 (*.pyc)|*.pyc|" \
           u" 垃圾邮件文件 (*.spam)|*.spam|" \
           "Egg file (*.egg)|*.egg|" \
           "All files (*.*)|*.*"


data_x={'dense_num':3,
        'cell_num':16,
        'activation':'relu',
        'train_steps':3000,
        'run_steps':1000,
        'model_name':'model_name',
        'algorithm_type':'dqn',
        'init_sate':[0,0,0,0],
        'action_space':2,

}

def get_config(configfile):
    cf=configparser.ConfigParser()
    cf.read(configfile)
    _conf_ints=[(key,int(value)) for key,value in cf.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in cf.items('strings')]
    return dict(_conf_ints+_conf_strings)

class RewardThread(threading.Thread):

    def __init__(self, parent,env):
        """

        """
        super(RewardThread, self).__init__()
        self.parent = parent
        self.env=env
        self._stop_event = threading.Event()
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

    def stop(self):
        self._stop_event.set()

class TrainThread(threading.Thread):

    def __init__(self, parent,client):
        super(TrainThread, self).__init__()
        self.parent = parent
        self.client=client
        self._stop_event = threading.Event()
        self.setDaemon(True)
    def run(self):

        self.client.train()

    def stop(self):
        self._stop_event.set()

class RunThread(threading.Thread):

    def __init__(self, parent,client):
        super(RunThread, self).__init__()
        self.parent = parent
        self.client=client
        self._stop_event = threading.Event()
        self.setDaemon(True)
    def run(self):

        self.client.run_model()

    def stop(self):
        self._stop_event.set()



class AppUI(wx.Panel):
        def __init__(self, parent):
            wx.Panel.__init__(self, parent)

            self.threads=[]

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
            self.logger = wx.TextCtrl(self, pos=(600, 20), size=(200, 350),
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


            self.insure = wx.StaticText(self, label="请选择Gym游戏", pos=(20, 180))
            grid.Add(self.insure, pos=(7, 0), span=(1, 2))


            radioList = ['CartPole', 'MountCar']
            rb = wx.RadioBox(self, label="", pos=(20, 210),
                             choices=radioList, majorDimension=3,
                             style=wx.RA_SPECIFY_COLS)
            grid.Add(rb, pos=(8, 0), span=(1, 2))
            self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, rb)
            # 运行按钮
            button_1 = wx.Button(self, label="训练", pos=(200, 325))
            self.Bind(wx.EVT_BUTTON, self.on_click_start, button_1)
            grid.Add(button_1, pos=(9, 0))

            button_2 = wx.Button(self, label="停止", pos=(200, 350))
            self.Bind(wx.EVT_BUTTON, self.on_click_stop, button_2)
            grid.Add(button_2, pos=(9, 1))

            hSizer.Add(grid, 0, wx.ALL, 5)
            hSizer.Add(self.logger)
            mainSizer.Add(hSizer, 0, wx.ALL, 5)

            self.SetSizerAndFit(mainSizer)

        def update_train_log(self):
            self.logger.AppendText('----------------------\n')
            self.logger.AppendText('|Train_steps  | %d\n' % self.step_data)
            self.logger.AppendText('|Train_reward | %d\n' % self.reward_data)

        def EvtRadioBox(self, event):

            if event.GetInt()==0:

                self.dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = self.dummy_env()
            elif event.GetInt()==1:
                dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = dummy_env()

        def EvtComboBox1(self, event):
            if event.GetString() == 'relu':
                data_x['activation'] = 'dqn'
            elif event.GetString() == 'sigmod':
                data_x['activation'] = 'sigmod'

            elif event.GetString() == 'softmax':
                data_x['activation'] = 'softmax'

        def EvtComboBox2(self, event):
            if event.GetString()=='dqn':
                data_x['algorithm_type']='dqn'
            elif event.GetString()=='ddpg':
                data_x['algorithm_type'] = 'ddpg'

            elif event.GetString()=='ppo':
                data_x['algorithm_type'] = 'ppo'




        #

        def EvtText1(self, event):
            data_x['model_name']=event.GetString()

        def EvtText2(self, event):
            data_x['dense_num']=int(event.GetString())

        def EvtText3(self, event):
            data_x['cell_num'] = int(event.GetString())

        def EvtText4(self, event):
            data_x['train_steps'] = int(event.GetString())








        def on_click_start(self,event):
            self.env = client(dense_num=data_x['dense_num'], cell_num=data_x['cell_num'], activation=data_x['activation'], train_steps=data_x['train_steps'],
                       dummy_env=self.dummy_env, model_name=data_x['model_name'], algorithm_type=data_x['algorithm_type'])


            self.progress_1 = RewardThread(self,self.env)
            self.progress_1.start()
            self.threads.append(self.progress_1)
            self.train = TrainThread(self,self.env)
            self.train.start()
            self.threads.append(self.train)

        def on_click_stop(self, event):
            for i in range(len(self.threads)):
               self.threads[i].stop
            self.threads=[]




class AppUI_1(wx.Panel):
        def __init__(self, parent):
            wx.Panel.__init__(self, parent)

            # 创建一些sizer
            self.client = None
            self.dummy_env = None
            self.step_data = 0
            self.reward_data = 0
            self.threads = []

            mainSizer = wx.BoxSizer(wx.VERTICAL)
            grid = wx.GridBagSizer(vgap=7, hgap=7)  # 虚拟网格sizer，此处指定了行和列之间的间隙
            hSizer = wx.BoxSizer(wx.HORIZONTAL)

            self.quote = wx.StaticText(self, label="", pos=(20, 20))
            grid.Add(self.quote, pos=(0, 0))

            # 展示训练过程的rewards
            self.logger = wx.TextCtrl(self, pos=(600, 20), size=(200, 350),
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
            self.sampleList = ['relu', 'sigmod', 'softmax']
            self.lblhear = wx.StaticText(self, label="神经网络激活函数", pos=(20, 70))
            grid.Add(self.lblhear, pos=(5, 0))
            self.edithear = wx.ComboBox(self, pos=(150, 70), size=(95, -1), choices=self.sampleList,
                                        style=wx.CB_DROPDOWN)
            grid.Add(self.edithear, pos=(5, 1))
            self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox1, self.edithear)

            # 组合框组件
            self.sampleList = ['dqn', 'ddpg', 'ppo']
            self.lblhear = wx.StaticText(self, label="选择强化学习算法", pos=(20, 70))
            grid.Add(self.lblhear, pos=(6, 0))
            self.edithear = wx.ComboBox(self, pos=(150, 70), size=(95, -1), choices=self.sampleList,
                                        style=wx.CB_DROPDOWN)
            grid.Add(self.edithear, pos=(6, 1))
            self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox2, self.edithear)

            self.lblname5 = wx.StaticText(self, label="URL：", pos=(30, 60))
            grid.Add(self.lblname5, pos=(7, 0))
            self.editname5 = wx.TextCtrl(self, value="请输入外部环境调用URL", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname5, pos=(7, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText5, self.editname5)

            self.lblname6 = wx.StaticText(self, label="初始化状态：", pos=(30, 60))
            grid.Add(self.lblname6, pos=(8, 0))
            self.editname6 = wx.TextCtrl(self, value="外部环境的初始化状态", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname6, pos=(8, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText6, self.editname6)

            self.lblname7 = wx.StaticText(self, label="action中包含的元素数量：", pos=(30, 60))
            grid.Add(self.lblname7, pos=(9, 0))
            self.editname7 = wx.TextCtrl(self, value="请填入正整数", pos=(140, 60), size=(140, -1))
            grid.Add(self.editname7, pos=(9, 1))
            self.Bind(wx.EVT_TEXT, self.EvtText7, self.editname7)

            # 运行按钮
            button_1 = wx.Button(self, label="训练", pos=(200, 325))
            self.Bind(wx.EVT_BUTTON, self.on_click_start, button_1)
            grid.Add(button_1, pos=(10, 0))

            button_2 = wx.Button(self, label="训练", pos=(200, 350))
            self.Bind(wx.EVT_BUTTON, self.on_click_stop, button_2)
            grid.Add(button_2, pos=(10, 1))

            hSizer.Add(grid, 0, wx.ALL, 5)
            hSizer.Add(self.logger)
            mainSizer.Add(hSizer, 0, wx.ALL, 5)
            self.SetSizerAndFit(mainSizer)

        def update_train_log(self):
            self.logger.AppendText('----------------------\n')
            self.logger.AppendText('|Train_steps  | %d\n' % self.step_data)
            self.logger.AppendText('|Train_reward| %d\n' % self.reward_data)

        def EvtRadioBox(self, event):

            if event.GetInt() == 0:

                self.dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = self.dummy_env()
            elif event.GetInt() == 1:
                dummy_env = lambda: gym.make('CartPole-v0').unwrapped
                self.dummy_env = dummy_env()

        def EvtComboBox1(self, event):
            if event.GetString() == 'relu':
                data_x['activation'] = 'dqn'
            elif event.GetString() == 'sigmod':
                data_x['activation'] = 'sigmod'

            elif event.GetString() == 'softmax':
                data_x['activation'] = 'softmax'

        def EvtComboBox2(self, event):
            if event.GetString() == 'dqn':
                data_x['algorithm_type'] = 'dqn'
            elif event.GetString() == 'ddpg':
                data_x['algorithm_type'] = 'ddpg'

            elif event.GetString() == 'ppo':
                data_x['algorithm_type'] = 'ppo'

        def EvtText1(self, event):
            data_x['model_name'] = event.GetString()

        def EvtText2(self, event):
            data_x['dense_num'] = int(event.GetString())

        def EvtText3(self, event):
            data_x['cell_num'] = int(event.GetString())

        def EvtText4(self, event):
            data_x['train_steps'] = int(event.GetString())

        def EvtText5(self, event):
            data_x['url'] = event.GetString()

        def EvtText6(self, event):
            a = []

            init_state = event.GetString().split(',')
            for i in init_state:
                a.append(float(i))
                print(a)
            data_x['init_state'] = a
            print(data_x['init_state'])

        def EvtText7(self, event):
            data_x['action_space'] = int(event.GetString())

        def on_click_start(self, event):
           self.dummy_env=Proxy(url=data_x['url'],init_state=data_x['init_state'])

           self.env = client(dense_num=data_x['dense_num'], cell_num=data_x['cell_num'],
                          activation=data_x['activation'], train_steps=data_x['train_steps'],
                          dummy_env=self.dummy_env, action_space=data_x['action_space'],model_name=data_x['model_name'],
                          algorithm_type=data_x['algorithm_type'])

           self.progress_1 = RewardThread(self, self.env)
           self.progress_1.start()
           self.threads.append(self.progress_1)
           self.train = TrainThread(self, self.env)
           self.train.start()
           self.threads.append(self.train)

        def on_click_stop(self, event):
          for i in range(len(self.threads)):
            self.threads[i].Destroy()
          self.threads = []



class AppUI_2(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # 创建一些sizer
        self.client = None
        self.dummy_env = None
        self.step_data = 0
        self.reward_data = 0
        self.threads=[]

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.GridBagSizer(vgap=7, hgap=7)  # 虚拟网格sizer，此处指定了行和列之间的间隙
        hSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.quote = wx.StaticText(self, label="", pos=(20, 20))
        grid.Add(self.quote, pos=(0, 0))

        # 展示训练过程的rewards
        self.logger = wx.TextCtrl(self, pos=(600, 20), size=(200, 350),
                                  style=wx.TE_MULTILINE | wx.TE_READONLY)

        # 编辑组件
        self.lblname1 = wx.StaticText(self, label="选择模型文件：", pos=(20, 60))
        grid.Add(self.lblname1, pos=(1, 0))
        button_3 = wx.Button(self, label="浏览", pos=(200, 325))
        self.Bind(wx.EVT_BUTTON, self.EvtText1, button_3)
        grid.Add(button_3, pos=(1, 1))



        self.lblname2 = wx.StaticText(self, label="选择模型超惨配置文件：", pos=(20, 60))
        grid.Add(self.lblname2, pos=(2, 0))
        button_4 = wx.Button(self, label="浏览", pos=(200, 325))
        self.Bind(wx.EVT_BUTTON, self.EvtText2, button_4)
        grid.Add(button_4, pos=(2, 1))



        self.lblname3 = wx.StaticText(self, label="模型运行步数", pos=(30, 60))
        grid.Add(self.lblname3, pos=(4, 0))
        self.editname3 = wx.TextCtrl(self, value="请输入正整数,输入-1则表示永远运行", pos=(140, 60), size=(140, -1))
        grid.Add(self.editname3, pos=(4, 1))
        self.Bind(wx.EVT_TEXT, self.EvtText3, self.editname3)


        self.lblname4 = wx.StaticText(self, label="URL：", pos=(30, 60))
        grid.Add(self.lblname4, pos=(7,0))
        self.editname4 = wx.TextCtrl(self, value="请输入外部环境调用URL", pos=(140, 60), size=(140, -1))
        grid.Add(self.editname4, pos=(7,1))
        self.Bind(wx.EVT_TEXT, self.EvtText4, self.editname4)

        self.lblname5 = wx.StaticText(self, label="初始化状态：", pos=(30, 60))
        grid.Add(self.lblname5, pos=(8, 0))
        self.editname5 = wx.TextCtrl(self, value="外部环境的初始化状态", pos=(140, 60), size=(140, -1))
        grid.Add(self.editname5, pos=(8, 1))
        self.Bind(wx.EVT_TEXT, self.EvtText5, self.editname5)


        # 运行按钮
        button_1 = wx.Button(self, label="运行模型", pos=(200, 325))
        self.Bind(wx.EVT_BUTTON, self.on_click_start, button_1)
        grid.Add(button_1, pos=(10, 0))

        button_2 = wx.Button(self, label="停止", pos=(200, 350))
        self.Bind(wx.EVT_BUTTON, self.on_click_stop, button_2)
        grid.Add(button_2, pos=(10, 1))

        hSizer.Add(grid, 0, wx.ALL, 5)
        hSizer.Add(self.logger)
        mainSizer.Add(hSizer, 0, wx.ALL, 5)
        self.SetSizerAndFit(mainSizer)

    def update_train_log(self):
        self.logger.AppendText('----------------------\n')
        self.logger.AppendText('|run_steps  | %d\n' % self.step_data)
        self.logger.AppendText('|run_reward| %d\n' % self.reward_data)
    def EvtText1(self, event):

        dlg = wx.FileDialog(self, message=u"选择文件",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=wildcard,
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
           data_x['model_name'] = dlg.GetFilename()
           print(data_x['model_name'])
        dlg.Destroy()
    def EvtText2(self, event):
        dlg = wx.FileDialog(self, message=u"选择文件",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=wildcard,
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
           filepath = dlg.GetPath()
           getconfig=get_config(filepath)
           data_x['cell_num']=getconfig['cell_num']
           data_x['dense_num']=getconfig['dense_num']
           data_x['activation']=getconfig['activation']
           data_x['algorithm_type']=getconfig['algorithm_type']
           data_x['action_space']=getconfig['action_space']
           dlg.Destroy()

    def EvtText3(self, event):
        data_x['run_steps'] = int(event.GetString())
        print(data_x['run_steps'])
    def EvtText4(self, event):
        data_x['url'] = event.GetString()

    def EvtText5(self, event):
        a=[]

        init_state=event.GetString().split(',')
        for i in init_state:
             i.strip('')
             a.append(float(i))
        print(a)
        data_x['init_state']=a

    def on_click_start(self, event):
        self.dummy_env=Proxy(url=data_x['url'],init_state=data_x['init_state'])
        self.env = client(dense_num=data_x['dense_num'], cell_num=data_x['cell_num'],
                          activation=data_x['activation'], train_steps=data_x['train_steps'],run_steps=data_x['run_steps'],
                          dummy_env=self.dummy_env, action_space=data_x['action_space'], model_name=data_x['model_name'],
                          algorithm_type=data_x['algorithm_type'])

        self.progress_1 = RewardThread(self, self.env)
        self.progress_1.start()
        self.threads.append(self.progress_1)
        self.run = RunThread(self, self.env)
        self.run.start()
        self.threads.append(self.run)

    def on_click_stop(self, event):
        for i in range(len(self.threads)):
            self.threads[i].Destroy()
        self.threads = []

if __name__ == "__main__":

        app = wx.App(False)
        frame = wx.Frame(None, title="General项目配置页面", size=(600, 400))
        nb = wx.Notebook(frame)

        nb.AddPage(AppUI(nb), "gym环境训练界面")
        nb.AddPage(AppUI_1(nb), "外部真实环境训练页面")
        nb.AddPage(AppUI_2(nb), "外部真实环境运行界面")
        frame.Show()
        app.MainLoop()



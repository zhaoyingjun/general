# 前言

&emsp;&emsp; 一直以来我都在思考如何将强化学习低门槛的引入到工业领域，现在我终于找到了实现这一目标的途径，那就是自己写一个强化学习应用编程框架。在研究了国外一个优秀的强化学框架（huskarl)之后，我决定基于该框架来写一个对国内开发者更友好的强化学习应用编程框架。框架取名general(将军），因为将军都是经过身经百战，在一次次的生死场景中训练成长出来的，广义上是一个强化学习的过程。
# general特性以及可视化演示
## general特性
&emsp;&emsp;general项目原计划至少实现四个方面的特性：支持可视化操作、集成主流强化学习算法、支持非gym环境交互、提供工业应用项目案例集，在1.0版本中实现了前三个特性的支持，工业应用项目案例集需要随着项目的推广和实践来积累。

 ### 支持可视化操作 
&emsp;&emsp;对于编程能力稍弱或者对于强化学习初学者来说，直接上手敲代码来实现强化学习是有困难的。如果能够通过界面配置就可以实现一个强化学习应用并能够直观的感受到其训练的过程，那么可以极大的提高初学者的兴趣和信心。因此，我用wxpython进行了Gui的开发，虽然界面做的比较丑但是基本上实现了想要达到的目的。在可视化模块，还有一些欠缺优化的部分，会在后面的迭代中持续进行更新，也欢迎wxpython的高手加入一起维护。可视化配置界面如图1，完全可视化配置后就可以搭建一个强化学习应用。
![图1:general配置页面](https://images.gitbook.cn/cc7b1450-5ada-11ea-aae1-d9c8e5f0a61f)
 ### 集成主流强化学习算法
   &emsp;&emsp;按照项目设计，在general中会集成主流的强化学习算法包括DQN、DDPG、PPO等，同时会关注强化学习的研究动态，及时将最新的强化学习算法进行集成。集成主流强化学习算法的目的降低强化学习算法应用的门槛，因此会对实现的过程和原理进行详细中文代码注释，以便让更多的人理解和掌握强化学习算法原理和实现过程。目前项目实现了DQN强化学习算法，具体的介绍会放在第三章节中。
   
     - [x] DQN:代码实现和详细注释，原理图对照讲解。
     - [ ] DDPG：待实现
     - [ ] PPO：待实现
     - [ ] ...

 ###  支持非gym环境交互
   &emsp;&emsp;当前的强化学习在训练过程中主流的是使用gym模拟实际的真实的环境进行交互，但是在工业生产有太多的场景是没有办法或者需要付出很大的成本才能进行抽象和模拟的。因此，在本框架中加了与非gym环境交互的模块，能够通过http或者grpc与真实环境交互。当然这里说的与真实环境交互是受控的交互，就是通过数据控制和真实环境的措施来避免训练过程造成实际的损失和危害，阿里每年双十一的全链路线上压测就是这样的思路，通过数据流量控制保证测试数据不会影响正常的交易数据。如果你想要尝试通过http的方式来与非gym环境交互，那么可以通过可视化界面配置项目或者直接调用client接口就可以。非gym环境交互配置如图2：
       ![非gym环境交互项目配置](https://images.gitbook.cn/bba08570-5af8-11ea-a8f4-7b4742511f3f)
 ###  工业应用项目集
&emsp;&emsp;当前强化学习最缺的不是前沿的理论研究，而是在工业领域应用的实践。现在很多很先进的算法的应用往往都是在与游戏环境的交互，哪怕是将DQN应用到工业环境中都是非常有价值的。因此我在这里也向各位读者征集在工作中遇到的难题，你有应用场景和数据，我有算法框架和编程技术，那么我们就可以一起搞点事情出来。如果能够搞成了，算法和代码你免费拿走，案例留下就可以。
 
     - [ ] 强化学习在燃气轮机自动调节优化中的应用：待应用实践
     - [ ] 强化学习在量化交易中的应用：尽管实用价值不大，但是后面会实现一下，因为环境和数据都比较充足。
     - [ ] ...   

## general可视化配置演示

 ###  gym交互环境配置演示
  &emsp;&emsp;在本框架中集成了gym，因此可以直接通过可视化界面完成项目的配置。   
  #### gym环境项目配置    
![gym环境项目配置](https://images.gitbook.cn/12009330-5b6a-11ea-9cd5-8967f4762932) 
#### gym交互环境训练过程
![gym交互环境训练](https://images.gitbook.cn/81555100-5b68-11ea-9cd5-8967f4762932)

  ###  非gym交互环境配置演示
&emsp;&emsp;在非gym交互环境下，我们采用http的方式与环境进行交互，因此需要先配置神经网络模型以及训练的超参数，然后配置环境交付url和环境的初始化状态。在本示例中，我使用gym的游戏后台写了一个模拟服务（urldemo.py），因此在使用前需要先启动该模拟服务。    
#### 配置神经网络和训练的超参数 
![配置神经网络和训练的超参](https://images.gitbook.cn/b57bd760-5b68-11ea-93b5-3993d9eab61e)   
#### 启动模拟服务    
![启动模拟服务urldemo.py](https://images.gitbook.cn/8a6f2e50-5b68-11ea-bd6c-43eeac1b3938)   
#### 将服务地址配置到项目中   
![将服务地址配置到项目中](https://images.gitbook.cn/951d1920-5b68-11ea-9cd5-8967f4762932)    
#### 非gyn交互环境的训练过程   
![非gym交互环境的训练过程](https://images.gitbook.cn/ab207960-5b68-11ea-bd6c-43eeac1b3938)
# general项目结构和模块功能介绍

## general项目结构

 ###  项目工程结构
&emsp;&emsp;general项目工程架构采用我一直推荐的“文件与文件夹”结构，就是基础稳定的文件放入文件中，动态的需要调试的放在根目录下。在general文件中包含了项目的核心模块，这些模块包括core、memory、policy、trainer以及一个算法包。在项目的根目录下放了代理层的client（代理终端）、proxy（服务代理）、urldemo（模拟服务）以及展现层的gui（可视化展现）、dqn-catpole（命令行展现)。    
![项目工程结构](https://images.gitbook.cn/0ac01ee0-5b6c-11ea-bd6c-43eeac1b3938)

 ###  项目功能结构
&emsp;&emsp;general从功能逻辑上分为四层，分别是核心层、算法层、代理层、展现层。核心层主要是实现框架中的ABC(抽象基类）的定义以及核心模块的实现，算法层主要是DQN、DDPG等算法智能体的实现、代理层主要是对外部非gym环境的交互代理，展现层包括基于wxpython的界面化交互和命令行的交互示例。   

![general功能架构](https://images.gitbook.cn/c5e94460-5b7d-11ea-a695-8f4c079b036d)
## general功能介绍
 ###  核心层
   * **memory:** memory是用来存储在强化学习训练过程的记录，以便能够通过记忆回放的方式训练Q网络，可以类比成人类的大脑记忆存储区域，里面包含了训练记录的存储、读取方法以及记忆回放策略（包括纯随机回放、有限随机回放、优先级回放）。   
   *  **core:** core是一个定义智能体的抽象基类文件，在文件中定义了一个智能体在实现过程中所必须具备的属性，比如模型文件保存、训练记录存储、做出行为指令预测等，这些抽象基类方法如果在实例化一个智能体过程中没有被实现则会宝未实现错误。    
   *  **policy:** policy是动作指令评估选择策略，对于深度神经网络模型会根据输入的环境状态预测出多个可执行动作，但是我们需要按照一定的选择策略算法对这些动作进行评价选择，最终输出动作指令。当前版本中实现的选择算法策略算法包括贪婪算法、随机贪婪算法、高斯随机贪婪算法。之所以会有不同算法是为了改进纯贪婪算法带来的局部最优解的困扰，通过随机贪婪算法、高斯随机贪婪算法能够在一定程度能够缓解或者解决贪婪算法带来的问题。      
   * **trainer:** trainer可以理解为训练模拟器，是一个有单进程或者多进程构成的“沙箱”。智能体与外界环境的交互、智能体的训练、智能体的记录存储等等这些过程都是在这个沙箱中。trainer可以构建包括单进程单实例、单进程多实例、多进程多实例的训练沙箱，单进程单实例和单进程多实例实现和训练过程相对比较简单，但是多进程多实例涉及到进程间通信、rewards同步、sates同步等过程非常的复杂。trainer的进程间通信和多实例训练同步的实现没有使用tensorflow的分布式多进程训练机制，而是采用消息队列基于python控制流来实现的。在后面的版本中会进行重构，采用tensorflow的架构和机制。

  ###  算法层
   * **DQN:** 当前在1.0版本中只实现了DQN算法智能体，不过根据core对算法智能的定义每个算法智能体中包含的方法都是一样的。在DQN智能体中包含loss函数定义、神经网络模型编译、train训练方法、save模型文件存储、push记录存储、act执行指令预测和选择。
  ###  代理层

   * **client:** client其实是可视化模块的后端服务，其主要作用是根据在gui可视化界面的超参配置构建一个训练实例并进行训练，将训练过程数据返回给可视化界面进行动态展示。在client中包括模型的定义、模型的训练、模型的保存、模型的测试等模块，实现从训练到测试的全过程。
   * **proxy:** proxy是一个服务代理模块，因为我们在对接非gym环境时需要通过http或者grpc的方式来进行，因此需要在服务代理模块中定义一些数据传输、数据处理的方法。目前在该模块中只实现了http的服务代理，后续版本中会增加grpc的代理。
   * **urldemo:** urldemo一个非gym环境模拟模块，也可以作为与非gym环境进行交互的客户端。在当前模块中，我们使用gym的cartpole游戏模拟作为外部环境的模拟，urldemo中实现了在环境本地端与环境的数据交互和调用，这也模拟在正常情况的场景。urldemo模块中包含对htrp传输数据的获取和序列化处理，对环境接口的交互和调用、对环境返回数据的json化处理以及返回。

  ###  展现层
   * **gui:** gui完全是基于wxpython编写的一个可视化模块，包含两个页面分别是gym环境配置页面和非gym环境配置页面，以实现对gym环境训练实例的配置和非gym环境训练实例的配置。在当前的模块中，受限于对wxpython的使用还不够娴熟，因此功能实现还不够丰富。在后续的版本中会对可视化页面进行改版以更加的符合工业应用使用体验和功能要求。比如后面会增加功能导航、训练模型实例的管理等功能，在页面的布局也会进行优化。
   * **dqn-cartpole:** dqn-cartpole是一个示例模块，展示如何使用general框架的api来完成自己所需的强化学习应用的编程。

# 使用说明

## 安装依赖

```
python setup.py install
```
## 运行可视化配置

```
pythonw gui.py
```

## 使用dqn-cartpole示例

```
pythonw dqn-cartpole
```

# 使用框架实现一个dqn-cartpole应用教程 

## 需要实现的功能模块
&emsp;&emsp;在本示例中，使用DQN网络和gym的托扁担游戏来作为示例，同时我们还会将gym的托扁担游戏来作为真实环境来演示如何通过http的方式与真实环境交互。我们需要实现以下的功能模块以搭建一个从训练和测试基本完整的强化学习应用：
 
 - 检验模型文件夹model_dir是否存在。
 - 托扁担（CartPole）游戏环境搭建。
 - 深度神经网络模型构建。
 - rewards反馈图形化展示。
 - 训练函数，如果是多次训练，则每次的新的训练是在之前训练成果的基础进行训练。
 - 模型测试函数，测试检验模型的训练效果。
 
## 代码实现详细过程
 ### 第一步需要导入各种依赖
```
import tensorflow as tf
import matplotlib
matplotlib.use("macOSX")#在使用macOSX系统时需要该行
import matplotlib.pyplot as plt
import gym
import general as gr
import os
```
 ### 第二步初始化gym环境
```
#初始化gym环境，使用CartPole-v0环境，就是托扁担游戏
create_env = lambda: gym.make('CartPole-v0').unwrapped
dummy_env = create_env()
```
 ### 第三步检验模型文件夹是否存在，如果不存在自动创建

```
if not os.path.exists("model_dir"):
      os.makedirs("model_dir")
```
 ### 第四步构建网络模型
&emsp;&emsp;我们使用TensorFlow2.0中的高阶API Sequential来构建神经网络模型，Sequential既以数组的方式来搭建网络模型也可以使用add方法搭建网络模型。下面代码示例是以数组的方式来搭建神经网络模型。
```
def create_model():
      # 我们搭建的神经网络模型一共三层，每层16个神经元，使用relu作为激活函数。
   model = tf.keras.Sequential([
         tf.keras.layers.Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
         tf.keras.layers.Dense(16, activation='relu'),
         tf.keras.layers.Dense(16, activation='relu'),
      ])
   return model
```
 ### 第五步我们使用matplotlib来实现对反馈reward的图形化展示
 
```
#定义反馈画图函数，这是为了能够图形化展示训练过程中rewards的变化走势，rewards是用来反馈对智能体的行为的评价。
def plot_rewards(episode_rewards, episode_steps, done=False):
     #初始化一块画布
      plt.clf()
      #设置XY坐标轴名称
      plt.xlabel('Step')
      plt.ylabel('Reward')
      #将反馈数据和训练步数全部画在画布中
      for ed, steps in zip(episode_rewards, episode_steps):
         plt.plot(steps, ed)
      plt.show() if done else plt.pause(0.001)
```
 ### 第六步我们定义一个训练函数来循环的训练模型

```
def train():
   #初始化神经网络模型
   model=create_model()
   #将定义好的网络作为参数传入general框架的API中，构成一个完成DQN 智能体，用于接下来的强化学习训练。
   agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=2)
   cpkt=tf.io.gfile.listdir("model_dir")
   if cpkt:
      agent.model.load_weights("model_dir/dqn.h5")
   #将智能体和gym环境放入训练器中开始训练深度神经网络模型
   tra = gr.Trainer(dummy_env, agent)
   tra.train(max_steps=3000, visualize=True, plot=plot_rewards)
   agent.save(filename='model_dir/dqn.h5',overwrite=True,save_format='h5')
```

 ### 第七步我们定义一个测试函数来检验模型训练的效果

```
def test():
   #初始化神经网络模型
    model=create_model()
    #将定义好的网络作为参数传入general框架的API中，构建一个含有DQN神经网络的智能体。
    agent = gr.DQN(model, actions=dummy_env.action_space.n, nsteps=2)
    #将之前训练的模型参数导入的新初始化的神经网络中
    agent.model.load_weights("model_dir/dqn.h5")
    #将智能体和gym环境放入训练器中开始测试模型的效果
    tra = gr.Trainer(create_env, agent)
    tra.test(max_steps=1000)
```
 ### 第八步我们定义一个主函数并设置一些交互选项,可以实现训练模式与测试模式的切换。
    
```
if __name__ == '__main__':
print("请准确输入train或者test")
#获取键盘输入
   mode=input()
#如果是训练模式，则调用训练函数
   if mode=="train":
      train()
 #如果是测试模式，则调用测试函数
   elif mode=="test":
      test()
   else:
      print("请重新执行程序并准确输入train或者test")
```


## 最终效果展示
&emsp;&emsp;本示例的代码也会包含在general项目中，可以直接下载下来进行调试。
 

 ### 示例的启动如下图   
   

 ![启动示例](https://images.gitbook.cn/462df5d0-5b06-11ea-a695-8f4c079b036d)
 

 ### 模型训练过程如下图，随着训练的进行智能体越来越能够保持木杆的垂直状态。
 


![模型训练过程](https://images.gitbook.cn/666b9ab0-5b0a-11ea-b7a0-3967fecd90d4)

 ### 模型的训练效果如下图，可以看到智能体可以很好的将木杆保持垂直状态。    
 


![测试模型](https://images.gitbook.cn/b1809a10-5b09-11ea-9cd5-8967f4762932)


# 结语
&emsp;&emsp;当前的版本还比较的基础，很多设计的功能和特性还没有来得及实现，如果你发现了一个bug或者希望加入到贡献者中一起维护该项目，我是非常欢迎的。同时，如果你在工作中有需要使用强化学习来解决问题的场景，还是那句话:你有场景需求和数据，我有框架和技术，我们为什么不一起搞点事情。最后，如果你对于强化学习感兴趣，那么请关注该项目[github地址](https://github.com/zhaoyingjun/general)，并顺手给个star哈。

# 版本迭代路线图
## 1.1版本 
- [ ] DDPG、PPO算法实现    

- [ ] 可视化界面布局优化

- [ ] grpc代理模块实现

- [ ] 量化投资应用案例

## 1.2版本    

- [ ] pytorch版本发布

- [ ] 底层性能优化





## 使用许可

[MIT](LICENSE) @zhaoyingjun

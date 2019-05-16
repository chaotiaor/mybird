"""
使用pytorch重建之后的deep bird,改进了网络结构，
使得模型更加的简单，同时也提高了准确性。

"""


from torch import nn, optim
import torch
import cv2
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import time


Game = 'bird'               # the name of the game being played for log files
Actions = 2                 # number of valid actions
Gamma = 0.99                # decay rate of past observations
Observe = 100000.           # timesteps to observe before training
Explore = 2000000.          # frames over which to anneal epsilon
Final_epsilon = 0.0001      # final value of epsilon
Initial_epsilon = 0.0001    # starting value of epsilon
Replay_memory = 50000       # number of previous transitions to remember
Batch = 32                  # size of minibatch
Width = 80                  # image width
High = 80                   # image high
Channel = 4                 # image channel
FRAME_PER_ACTION = 1        # frame per action


class Net(nn.Module):
    """
    get the network

    结构如下：
    conv -> relu-> pool -> relu-> conv -> relu-> conv -> relu-> linear
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # self.liner1 = nn.Linear(2304, 256)
        # self.liner = nn.Linear(256, 2)
        self.liner = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        # x = self.pool(x)
        # x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        # x = self.pool(x)
        # print(x.shape)
        # x = self.relu(x)
        x = x.view(x.shape[0], -1)
        # x = self.liner1(x)
        out = self.liner(x)
        # out = torch.sum(torch.mul(out, a), 1)
        return out


class QLearn:
    """
    这个模块包括
    load        加载模型
    save        保存模型
    get_tensor  把numpy数组转化为tensor
    get_state   得到游戏的状态
    eval        验证模型
    decision    进行决策
    train       训练模型
    go          流程化训练，验证，决策模型。

    """

    def __init__(self, model, path):
        """
        self.net            实例化模型
        self.path           模型保存路径
        self.game_state     游戏状态
        self.batch_s_t      t时刻的图像像素的batch
        self.batch_s_t1     t1时刻的图像像素的batch
        self.batch_a_t      t时刻的行动的batch
        self.batch_r        t时刻对应的奖励的batch
        self.y_batch        t时刻对应的奖励 加上 最大的模型输出值 × 系数，也就是对应的q-value
        self.s_t            t时刻的图像像素
        self.loss_data      loss的数值
        self.readout_t      t时刻的模型输出
        self.r_t            t时刻对应的奖励
        self.s_t1           t1时刻的图像像素
        self.action_index   行动下标
        self.t              步数，记录运行多少步
        self.loss_function  损失函数
        self.optimizers     优化器
        self.epsilon        系数
        self.D              数据队列
        self.load(path)     加载模型
        self.observe        观察步数
        self.explore        探索步数
        self.cuda           是否使用cuda

        :param model:
        """
        self.net = model()
        self.path = path
        self.game_state = game.GameState()
        self.batch_s_t = np.zeros([Batch, Channel, Width, High])
        self.batch_s_t1 = np.zeros([Batch, Channel, Width, High])
        self.batch_a_t = np.zeros([Batch, Actions])
        self.batch_r = np.zeros([Batch])

        self.s_t = self.get_state()
        self.loss_data = 0
        self.readout_t = None
        self.r_t = None
        self.s_t1 = None
        self.y_batch = np.zeros([Batch])
        self.action_index = 0
        self.t = 0
        # self.death = 0
        self.loss_function = nn.MSELoss()
        self.optimizers = optim.Adam(params=self.net.parameters(),
                                lr=1e-8
                                )
        self.epsilon = Initial_epsilon
        self.D = deque()
        self.load(path)
        self.observe = self.t + Observe
        self.explore = self.t + Explore
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            self.net = self.net.cuda()
        print("begin time", time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime()))

    def load(self, path):
        """

        :return:
        """
        try:
            path = '/'.join([path, "{}.pth".format(Game)])
            files = torch.load(path)
            models_status = files['state_dict']
            t = files['t']
        except FileNotFoundError as e:
            print(e)
            return None
        else:
            self.net.load_state_dict(models_status)
            self.t = t
            return None

    def save(self, path):
        """

        :return:
        """
        files = {"state_dict": self.net.state_dict(),
                 "t": self.t
                 }
        path = '/'.join([self.path, "{}.pth".format(path)])
        torch.save(files, path)
        return None

    @staticmethod
    def get_tensor(data):
        _data = data.copy()
        _data = torch.from_numpy(_data)
        return _data.float()

    def get_state(self):
        do_nothing = np.zeros([Actions])
        do_nothing[0] = 1
        x_t, r_0, terminal = self.game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)),
                           cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255,
                                 cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        s_t = s_t[np.newaxis, :]
        return s_t

    def eval(self):
        """

        :return:
        """
        self.net.eval()
        minibatch = random.sample(self.D, Batch)

        # get the batch variables
        for index, d in enumerate(minibatch):
            self.batch_s_t[index] = d[0]
            self.batch_a_t[index] = d[1]
            self.batch_r[index] = d[2]
            self.batch_s_t1[index] = d[3]

        s_j1_batch = self.get_tensor(self.batch_s_t1)
        if self.cuda:
            # s_j1_batch = s_j1_batch.cuda()
            readout_j1_batch = np.array(
                self.net(s_j1_batch.cuda()).data.cpu())

        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                self.y_batch[i] = self.batch_r[i]
            else:
                self.y_batch[i] = (self.batch_r[i] +
                               Gamma * np.max(readout_j1_batch[i])
                               )


    def decision(self):
        """

        :return:
        """
        self.net.eval()
        a_t = np.zeros([Actions])
        s_t = self.get_tensor(self.s_t)
        if self.cuda:
            # s_t = s_t.cuda()
            self.readout_t = np.array(self.net(s_t.cuda()).data.cpu())

        if self.t % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                print("----------Random Action----------")
                self.action_index = random.randrange(Actions)
                a_t[self.action_index] = 1
            else:
                self.action_index = np.argmax(self.readout_t)
                a_t[self.action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        if self.epsilon > Final_epsilon and self.t > Observe:
            self.epsilon -= (Initial_epsilon - Final_epsilon) / Explore

        # run the selected action and
        # observe next state and reward
        x_t1_colored, self.r_t, terminal = self.game_state.frame_step(a_t)


        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1, 80, 80))
        x_t1 = x_t1[np.newaxis, :]
        self.s_t1 = np.append(x_t1, self.s_t[:, :3, :, :], axis=1)

        # store the transition in D
        self.D.append((s_t, a_t, self.r_t, self.s_t1, terminal))
        if len(self.D) > Replay_memory:
           self.D.popleft()

        return None


    def train(self):
        """

        :return:
        """

        self.net.train()
        s_t_batch = self.get_tensor(self.batch_s_t)
        y_batch = self.get_tensor(self.y_batch)
        a_batch = self.get_tensor(self.batch_a_t)

        if self.cuda:
            # s_t_batch = s_t_batch.cuda()
            # a_batch = a_batch.cuda()
            # y_batch = y_batch.cuda()
            outputs = self.net(s_t_batch.cuda())

        loss = self.loss_function(
            outputs.mul(a_batch.cuda()).sum(1), y_batch.cuda())
        self.loss_data = np.array(loss.data.cpu())

        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()

        return None

    def go(self):
        """

        :return:
        """
        while "flappy bird" != "angry bird":
            self.decision()
            if self.t > self.observe:
                self.eval()
                self.train()
                # print("loss_{}".format(self.loss_data))
            self.s_t = self.s_t1
            self.t += 1

            # save progress every 10000 iterations
            if self.t % 10000 == 0:
                self.save(Game + '{}'.format(self.t))

            # print info
            if self.t <= self.observe:
                state = "observe"
            elif (self.t > self.observe) and \
                    (self.t <= self.observe + self.explore):
                state = "explore"
            else:
                state = "train"

            if self.r_t == -1:
                print("TIMESTEP", self.t,
                      "/ STATE", state,
                      "/ EPSILON", self.epsilon,
                      "/ ACTION", self.action_index,
                      "/ REWARD", self.r_t,
                      "/ Q_MAX %e" % np.max(self.readout_t),
                      )
                print("/death time", time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime()))


def main(path):
    QLearn(Net, path).go()


if __name__ == "__main__":
    """
    """
    _path = './saved_networks/'
    main(_path)














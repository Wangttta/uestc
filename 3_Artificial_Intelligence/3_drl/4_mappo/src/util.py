import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime

"""
这里定义了当前项目运行时，所有可用的系统及命令行参数
"""
def get_args():
    parser = argparse.ArgumentParser("PPO & CartPole-v1 experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=3, help="Number of agent")
    parser.add_argument("--t", type=int, default=25, help="Number of steps for each episode")
    parser.add_argument("--n-episode", type=int, default=120000, help="Running episode")
    parser.add_argument("--n-train", type=int, default=15, help="Training times for each episode")
    parser.add_argument("--n-eval", type=int, default=3, help="Evaluate times (num of episode)")
    parser.add_argument("--n-eval-rate", type=int, default=200, help="Evaluate frequency (num of episode)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini-batch-size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip boundary")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--hidden-dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--logdir", type=str, default="../data", help="Default log directory")
    parser.add_argument("--log-rate", type=int, default=1000, help="Logging frequency of neural network and uav trajectory")
    args = parser.parse_args()
    return args


def log(text, condition=True):
    if condition:
        print(text)


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

class RunningMeanStd:

    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:

    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            x = np.array([x[agent] for agent in x.keys()])
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x



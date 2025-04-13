import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime


"""
定义当前项目运行时，所有可用的系统及命令行参数
"""
def get_args():
    parser = argparse.ArgumentParser("PPO & CartPole-v1 experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--m", type=int, default=100, help="Number of user device")
    parser.add_argument("--n", type=int, default=4, help="Number of agent")
    parser.add_argument("--t", type=int, default=500, help="Number of steps for each episode")

    parser.add_argument("--ue-x-min", type=float, default=200, help="")
    parser.add_argument("--ue-y-min", type=float, default=200, help="")
    parser.add_argument("--ue-z-min", type=float, default=0, help="")
    parser.add_argument("--ue-x-max", type=float, default=800, help="")
    parser.add_argument("--ue-y-max", type=float, default=800, help="")
    parser.add_argument("--ue-z-max", type=float, default=0, help="")
    parser.add_argument("--uav-x-min", type=float, default=0, help="")
    parser.add_argument("--uav-y-min", type=float, default=0, help="")
    parser.add_argument("--uav-z-min", type=float, default=30, help="")
    parser.add_argument("--uav-x-max", type=float, default=1000, help="")
    parser.add_argument("--uav-y-max", type=float, default=1000, help="")
    parser.add_argument("--uav-z-max", type=float, default=50, help="")
    parser.add_argument("--uav-f-min", type=int, default=10000000000, help="10GHz")
    parser.add_argument("--uav-f-max", type=int, default=10000000000, help="10GHz")
    parser.add_argument("--ue-f-min", type=int, default=1000000000, help="1GHz")
    parser.add_argument("--ue-f-max", type=int, default=5000000000, help="5GHz")
    parser.add_argument("--task-size-min", type=int, default=10485760, help="10MB")
    parser.add_argument("--task-size-max", type=int, default=20971520, help="20MB")
    parser.add_argument("--uav-angle", type=float, default=170, help="Max UAV covering angle (flare angle)")
    parser.add_argument("--g2a-noise", type=float, default=1e-8, help="Noise of G2A link (-50 dBm, watt)")
    parser.add_argument("--env-a", type=float, default=9.61, help="env constant for calculating LoS & NLoS probability")
    parser.add_argument("--env-b", type=float, default=0.16, help="env constant for calculating LoS & NLoS probability")
    parser.add_argument("--env-c", type=float, default=3e8, help="speed of light")
    parser.add_argument("--env-fr-c", type=float, default=1e9, help="carrier frequency")
    parser.add_argument("--env-eta-los", type=float, default=1, help="Excessive pass loss for LoS")
    parser.add_argument("--env-eta-nlos", type=float, default=20, help="Excessive pass loss for NLoS")
    parser.add_argument("--g2a-bandwidth", type=float, default=2e7, help="Bandwidth of G2A link (Hz)")

    parser.add_argument("--n-episode", type=int, default=200000, help="Running episode")
    parser.add_argument("--n-train", type=int, default=15, help="Training times for each episode")
    parser.add_argument("--n-eval", type=int, default=3, help="Evaluate times (num of episode)")
    parser.add_argument("--n-eval-rate", type=int, default=100, help="Evaluate frequency (num of episode)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini-batch-size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip boundary")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--hidden-dim", type=int, default=256, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--logdir", type=str, default="../data/runtime", help="Default log directory")
    parser.add_argument("--log-rate", type=int, default=2, help="Logging frequency of neural network and uav trajectory")
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


def minmax(np_arr, min_val, max_val):
    if min_val == max_val:
        max_val += 1e-9
    return (np_arr - min_val) / (max_val - min_val)


def calc_cover_radius(angle, height):
    return np.tan(angle * np.pi / 360) * height


class RunningMeanStd:

    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.S = np.zeros(shape)
        self.mean = np.zeros(shape)
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
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

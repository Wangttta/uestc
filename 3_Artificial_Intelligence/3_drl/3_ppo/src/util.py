import argparse

"""
这里定义了当前项目运行时，所有可用的系统及命令行参数
"""
def get_args():
    parser = argparse.ArgumentParser("PPO & CartPole-v1 experiment")
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    parser.add_argument("--n-episode", type=int, default=1000, help="Running episode")
    parser.add_argument("--n-train", type=int, default=10, help="Training times for each episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="LR of actor network")
    parser.add_argument("--lr-decay-rate", type=float, default=0.995, help="LR decay rate")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip boundary")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--dir-board", type=str, default="", help="Log directory of tensorboard")
    args = parser.parse_args()
    return args


def log(text, condition=True):
    if condition:
        print(text)
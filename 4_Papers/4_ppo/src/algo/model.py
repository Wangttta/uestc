import os
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.drl.ppo import PPO
from env.umec import UMEC
from common.utils import *


class Model:
    
    def __init__(self, args, env: UMEC):
        # 1. Initialize parameters, environment, and saving paths
        self.args, self.env = args, env
        args.root_path = os.path.join(args.root_path, datetime_str())
        # 1.1. 记录无人机轨迹图
        if args.dir_traj is not None:
            args.path_uav_traj = os.path.join(args.root_path, args.dir_traj)
            os.makedirs(args.path_uav_traj)
        # 1.2. 记录模型参数
        if args.dir_model is not None:
            args.path_model = os.path.join(args.root_path, args.dir_model)
            os.makedirs(args.path_model)
        # 1.3. 记录训练参数以及初始化 Tensorboard
        if args.dir_board is not None:
            args.path_board = os.path.join(args.root_path, args.dir_board)
            os.makedirs(args.path_board)
            self.writer = SummaryWriter(args.path_board)
        # 2. Create algorithm model
        self.drl_model = PPO(args=args)
        # 3. Statistics
        self.step = 0

    def run(self):
        for episode in range(self.args.n_episode):
            require_log = (episode % 10 == 0)
            # 4.1. Collect experience
            log("--------------------------------------------------", require_log)
            log(f"P1. Episode={episode}, collecting data ...", require_log)
            state, _ = self.env.reset()
            done, episode_reward, step = False, 0, 0
            while not done:
                action, log_prob = self.drl_model.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step_action(action)
                self.drl_model.buffer.save(state, action, log_prob, reward, done, truncated)
                state = next_state
                episode_reward += reward
                done = (done or truncated)
                step += 1
            self.env.render(path=os.path.join(self.args.path_uav_traj, f"UavTraj_{episode}"))
            self.writer.add_scalar('Reward', episode_reward, episode)
            log(f"P1. Episode={episode}, collect over, score={episode_reward}, step={step}", require_log)
            # 4.2. Training
            log(f"P2. Episode={episode}, training ...", require_log)
            self.drl_model.learn()
            self.writer.add_scalar('Loss', self.drl_model.loss, episode)
            log(f"P2. Episode={episode}, train over ...", require_log)
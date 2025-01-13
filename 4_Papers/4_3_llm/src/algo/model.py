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
        if args.dir_log is not None:
            args.path_log = os.path.join(args.root_path, args.dir_log)
            os.makedirs(args.path_log)
            self.writer = SummaryWriter(args.path_log)
        # 2. Create algorithm model
        self.state_dim = env.state_dim
        self.action_dim = np.array([env.N * 9, env.N * 3])  # 无人机轨迹这一动作需要两个维度参数：展开维度、输出维度
        self.drl_model = PPO(args=args, state_dim=env.state_dim, action_dim=self.action_dim)
        # 3. Statistics
        self.norm_mean = np.zeros(self.env.state_dim)
        self.norm_std = np.ones(self.env.state_dim)
        self.episode_counter, self.step_counter = 0, 0

    def __collect_experiences(self, epoch=0):
        epoch_reward = 0
        for episode in range(self.args.n_episode):
            done, episode_reward = False, 0
            while not done:
                # 1. Observe state
                state, mask = self.env.state()
                state = (state - self.norm_mean) / np.maximum(self.norm_std, 1e-6)  # State normalization
                # 2. Select actions
                value, action, log_prob = self.drl_model.select_action(state, mask)
                # 3. Step actions
                reward, next_state, done = self.env.step_action((action - 1) * 10)
                episode_reward += reward
                # 4. Save experience
                self.drl_model.buffer.store(state, action, mask, reward, done, value, log_prob)
                # 5. Update state and normalization parameters
                state = next_state
                self.step_counter += 1
                self.writer.add_scalar('Env/reward', reward, self.step_counter)
            episode_reward /= self.args.t
            epoch_reward += episode_reward
            print(f"P1. Epoch={epoch}, Episode={episode}, collect data, avg_episode_reward={episode_reward}")
            if self.args.path_uav_traj is not None:
                self.env.render(path=os.path.join(self.args.path_uav_traj, f"UavTraj_{epoch}_{episode}"))
            # TensorBoard
            self.episode_counter += 1
            self.writer.add_scalar('Env/episode_reward', episode_reward, self.episode_counter)
        return epoch_reward / self.args.n_episode

    def run(self):
        for i in range(self.args.n_epoch):
            print(f"--------------------------------------------------")
            print(f"P1. Epoch={i}, collecting data ...")
            self.env.reset()
            with torch.no_grad():
                avg_reward = self.__collect_experiences(epoch=i)
            print(f"P1. Epoch={i}, collect data over, avg_epoch_reward={avg_reward}")
            print(f"P2. Epoch={i}, training ...")
            self.drl_model.train(epoch=i, writer=self.writer)
            self.drl_model.save_model(path=os.path.join(self.args.path_model, f"Model_{i}"))
            self.norm_mean = np.tile(self.drl_model.buffer.filter()[0], self.state_dim)
            self.norm_std = np.tile(self.drl_model.buffer.filter()[1], self.state_dim)
            self.drl_model.buffer.clear()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.spaces import *

from common.utils import *


class UMEC:

    def __init__(self, args):
        # 1. Initialize parameters
        self.args = args
        self.M, self.N, self.T = args.m, args.n, args.t
        # 2. 初始化状态和动作空间
        # 2.1. 状态空间：每个用户的二维坐标 + 每个无人机的二维坐标（均为区域边界）
        low_state = np.array([self.args.x_min, self.args.y_min] * (self.M + self.N), dtype=np.float32)
        high_state = np.array([self.args.x_max, self.args.y_max] * (self.M + self.N), dtype=np.float32)
        self.observation_space = Box(low_state, high_state, dtype=np.float32)
        # 2.2. 连续动作空间：每个无人机的二维移动，每个维度是一个属于 [-1, 1] 的连续变量
        if args.continuous_actions:
            low_action = np.array([-1, -1] * self.N)
            high_action = np.array([1, 1] * self.N)
            self.action_space = Box(low_action, high_action, dtype=np.float32)
        # 2.3. 离散动作空间：每个无人机的二维移动，每个维度是一个属于 {-1, 0, 1} 的离散变量
        else:
            self.action_space = Discrete(9 * self.N)
        # 3. Initialize the UEs and UAVs
        self.ue_pos = self.__init_ue()
        self.uav_pos = self.__init_uav()
        self.trajectories, self._trajectories  = [], None
        self._ue_pos, self._uav_pos = np.copy(self.ue_pos), np.copy(self.uav_pos)
        self.max_dis = np.sum(np.linalg.norm(self.uav_pos[:, np.newaxis, :] - self.ue_pos[np.newaxis, :, :], axis=2))
        self.step_count = 0

    def __init_ue(self):
        # Type 1: Random
        # x = np.random.uniform(self.args.x_min, self.args.x_max, (self.M, 1))
        # y = np.random.uniform(self.args.y_min, self.args.y_max, (self.M, 1))
        # z = np.zeros((self.M, 1))
        # return np.concatenate((x, y, z), axis=1).astype(np.float32)
        # Type 2: Central
        off_x = (self.args.x_max - self.args.x_min) / 3
        off_y = (self.args.y_max - self.args.y_min) / 3
        x = np.random.uniform(self.args.x_min + off_x, self.args.x_max - off_x, (self.M, 1))
        y = np.random.uniform(self.args.y_min + off_y, self.args.y_max - off_y, (self.M, 1))
        return np.concatenate((x, y, np.zeros((self.M, 1))), axis=1).astype(np.float32)

    def __init_uav(self):
        # Type 1: Random
        # x = np.random.uniform(self.args.x_min, self.args.x_max, (self.N, 1))
        # y = np.random.uniform(self.args.y_min, self.args.y_max, (self.N, 1))
        # z = np.random.uniform(self.args.z_min, self.args.z_max, (self.N, 1))
        # return np.concatenate((x, y, z), axis=1).astype(np.float32)
        # Type 2: Fixed start position
        xoy = np.zeros((self.N, 2))
        z = np.ones((self.N, 1)) * self.args.z_min
        return np.concatenate((xoy, z), axis=1).astype(np.float32)

    def state(self):
        ue_x = min_max(self.ue_pos[:, 0], min_val=self.args.x_min, max_val=self.args.x_max)
        ue_y = min_max(self.ue_pos[:, 1], min_val=self.args.y_min, max_val=self.args.y_max)
        uav_x = min_max(self.uav_pos[:, 0], min_val=self.args.x_min, max_val=self.args.x_max)
        uav_y = min_max(self.uav_pos[:, 1], min_val=self.args.y_min, max_val=self.args.y_max)
        state = np.hstack((ue_x, ue_y, uav_x, uav_y)).astype(np.float32)
        return state

    def reset(self):
        self._trajectories = np.copy(self.trajectories)
        self.trajectories = []
        self.ue_pos, self.uav_pos = np.copy(self._ue_pos), np.copy(self._uav_pos)
        self.step_count = 0
        return self.state(), None

    def step_action(self, action):
        # 1. Save previous trajectory
        self.trajectories.append(np.copy(self.uav_pos))
        # 2. Execute action
        x = (action // 3 - 1) * 10
        y = (action % 3 - 1) * 10
        self.var_w = np.array([x, y, 0], dtype=np.float32)  # UAVs movement
        self.uav_pos += self.var_w
        penalty = 0
        for n in range(self.N):
            if self.uav_pos[n, 0] < self.args.x_min: penalty += 1
            if self.uav_pos[n, 0] > self.args.x_max: penalty += 1
            if self.uav_pos[n, 1] < self.args.y_min: penalty += 1
            if self.uav_pos[n, 1] > self.args.y_max: penalty += 1
        lower_bound = np.array([self.args.x_min, self.args.y_min, self.args.z_min])
        upper_bound = np.array([self.args.x_max, self.args.y_max, self.args.z_max])
        self.uav_pos = np.clip(self.uav_pos, lower_bound, upper_bound)
        # 3. Calculate reward
        reward, done = self.__calc_reward()
        reward -= penalty
        # 4. Next state
        self.step_count += 1
        truncate = (self.step_count == self.args.t)
        if done or truncate:
            self.reset()
        next_state = self.state()
        return next_state, reward, done, truncate, None
    
    def render(self, path=None):
        plt.clf()
        ax = plt.subplot(projection='3d')
        ax.set_title(self.args.env_name)
        colors = ['red', 'green', 'blue', 'yellow', 'black']
        for i in range(self.N):
            uav_traj = self._trajectories[:, i, :]
            ax.scatter(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], marker='p', c=colors[i], s=10)
            ax.scatter(uav_traj[0, 0], uav_traj[0, 1], uav_traj[0, 2], marker='^', c=colors[-1], s=20)
            center = [uav_traj[-1, 0], uav_traj[-1, 1], 0]
            radius = calc_cover_radius(angle=self.args.uav_angle, height=uav_traj[-1, 2])
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = np.ones_like(x) * center[2]
            ax.plot(x, y, z, c=colors[i], linewidth=1, markersize=5)
        ax.scatter(self.ue_pos[:, 0], self.ue_pos[:, 1], self.ue_pos[:, 2], c='b', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xmin=self.args.x_min, xmax=self.args.x_max)
        ax.set_ylim(ymin=self.args.y_min, ymax=self.args.y_max)
        ax.set_zlim(zmin=0, zmax=self.args.z_max)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def __calc_reward(self):
        distance = np.sum(np.linalg.norm(self.uav_pos[:, np.newaxis, :] - self.ue_pos[np.newaxis, :, :], axis=2))
        reward = - distance / self.max_dis
        done = (distance <= 100)
        return reward, done

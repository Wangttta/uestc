import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from util import *


class UMEC:

    def __init__(self, args) -> None:
        # 1. 基本参数
        self.args = args
        self.seed = args.seed
        self._np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(args.seed)))
        self.M, self.N, self.T = args.m, args.n, args.t
        self.t_loc, self.t_off, self.t_exe, self.t_resp = None, None, None, None
        self.ues = self.init_ues()
        self.uavs = self.init_uavs()
        self.tasks = self.generate_tasks()
        self._ues, self._uavs, self._tasks = np.copy(self.ues), np.copy(self.uavs), np.copy(self.tasks)
        # 2. 每个无人机的状态空间（同构智能体，维度均相同）
        # 2.1. 所有无人机的位置
        # 2.2. 所有用户的位置（部分观测，非观测范围内置空）
        # 2.3. 所有用户的任务信息（部分观测，非观测范围内置空）
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=np.array([args.uav_x_min, args.uav_y_min, args.uav_z_min] * self.N), high=np.array([args.uav_x_max, args.uav_y_max, args.uav_z_max] * self.N)),
            gym.spaces.Box(low=np.array([args.ue_x_min, args.ue_y_min, args.ue_z_min] * self.M), high=np.array([args.ue_x_max, args.ue_y_max, args.ue_z_max] * self.M)),
            gym.spaces.Box(low=np.array([args.task_size_min, args.ue_f_min] * self.M), high=np.array([args.task_size_max, args.ue_f_max] * self.M))
        ), seed=args.seed)
        # 3. 每个无人机的动作空间（同构智能体，维度均相同）
        # 3.1. 任务卸载决策（离散动作，对于每个智能体来说，有 M 个二进制变量表示卸载到该无人机的用户任务）
        # 3.2. 无人机移动（连续动作）
        # 3.3. 当前无人机的资源分配（连续动作）
        self.action_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(np.array([2] * self.M)),
            gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
        ), seed=args.seed)
        # 4. 每个智能体的状态维度和动作维度
        self.state_dim = 3 * self.N + 3 * self.M + 2 * self.M  # 全局状态空间维度
        self.obs_dim = self.state_dim   # 智能体的观测维度：数值与全局状态相同，但只包含部分有效数据（本质是部分观测）
        self.act_dim_dis = (self.M, 2)  # 智能体离散动作空间维度：M 个离散动作，每个动作有两种取值
        self.act_dim_con = 3            # 智能体连续动作空间维度：3 个连续动作
        # 5. 辅助数据结构
        self.step = 0
        self.trajectories, self._trajectories = np.empty((self.N, self.T, 3)), None
        self.dis_mat, self.link_mat, self.nearest_link_vec = None, None, None  # 用户到无人机的距离矩阵、用户与无人机的连接矩阵（是否在覆盖范围内）、每个用户的最近无人机向量
        self.off_set, self.off_path_mat = None, None  # 无人机的卸载用户集合、用户设备的卸载矩阵（包含多跳信息）

    def init_ues(self):
        x = self._np_random.uniform(self.args.ue_x_min, self.args.ue_x_max, self.M).reshape((-1, 1))
        y = self._np_random.uniform(self.args.ue_y_min, self.args.ue_y_max, self.M).reshape((-1, 1))
        z = np.zeros((self.M, 1))
        ues_cpu = self._np_random.uniform(self.args.ue_f_min, self.args.ue_f_max, self.M).reshape((-1, 1))
        return np.concatenate((x, y, z, ues_cpu), axis=1).astype(np.float32)

    def init_uavs(self):
        # x = self._np_random.uniform(self.args.uav_x_min, self.args.uav_x_max, self.N).reshape((-1, 1))
        # y = self._np_random.uniform(self.args.uav_y_min, self.args.uav_y_max, self.N).reshape((-1, 1))
        # z = self._np_random.uniform(self.args.uav_z_min, self.args.uav_z_max, self.N).reshape((-1, 1))
        x = np.array([self.args.uav_x_min, self.args.uav_x_min, self.args.uav_x_max, self.args.uav_x_max]).reshape((-1, 1))
        y = np.array([self.args.uav_y_min, self.args.uav_y_max, self.args.uav_y_min, self.args.uav_y_max]).reshape((-1, 1))
        z = np.array([self.args.uav_z_max, self.args.uav_z_max, self.args.uav_z_max, self.args.uav_z_max]).reshape((-1, 1))
        return np.concatenate((x, y, z), axis=1).astype(np.float32)

    def generate_tasks(self):
        """
        Each task is defined as a 2 tuple: (task_size, task_cpu_cycles, max_delay), 
        - task_size: total bytes in transmission (MB, acturally)
        - task_cpu_cycles: total cpu cycles required for calculating
        - max_delay: maximum allowed delay threshold
        """
        tasks_size = self._np_random.uniform(self.args.task_size_min, self.args.task_size_max, size=self.M).astype(np.float32)
        tasks_cpu = (tasks_size * 100).astype(np.float32)
        tasks_delay = tasks_cpu / self.ues[:, 3]
        self.t_loc = tasks_delay.copy()
        return np.concatenate((tasks_size.reshape(-1, 1), tasks_cpu.reshape(-1, 1), tasks_delay.reshape(-1, 1)), axis=1).astype(np.float32)

    def reset(self):
        self._np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))
        self._trajectories = np.copy(self.trajectories)
        self.trajectories = np.empty((self.N, self.T, 3))
        self.step = 0
        self.ues, self.uavs, self.tasks = np.copy(self._ues), np.copy(self._uavs), np.copy(self._tasks)
        return self.state()

    def state(self):
        # 1. 状态信息（全局状态、每个智能体的部分观测）
        uav_pos, ue_pos = self.uavs[:, :3], self.ues[:, :3]
        cover_radius = calc_cover_radius(self.args.uav_angle, uav_pos[:, 2]).reshape(1, -1)
        distance_mat = np.linalg.norm(ue_pos[:, np.newaxis, :] - uav_pos[np.newaxis, :, :], axis=2)
        cover_mat = distance_mat <= cover_radius
        uav_x = minmax(self.uavs[:, 0], min_val=self.args.uav_x_min, max_val=self.args.uav_x_max)
        uav_y = minmax(self.uavs[:, 1], min_val=self.args.uav_y_min, max_val=self.args.uav_y_max)
        uav_z = minmax(self.uavs[:, 2], min_val=self.args.uav_z_min, max_val=self.args.uav_z_max)
        ue_x = minmax(self.ues[:, 0], min_val=self.args.ue_x_min, max_val=self.args.ue_x_max)
        ue_y = minmax(self.ues[:, 1], min_val=self.args.ue_y_min, max_val=self.args.ue_y_max)
        ue_z = minmax(self.ues[:, 2], min_val=self.args.ue_z_min, max_val=self.args.ue_z_max)
        task_size = minmax(self.tasks[:, 0], min_val=self.args.task_size_min, max_val=self.args.task_size_max)
        task_cpu = minmax(self.tasks[:, 1], min_val=self.args.task_size_min * 100, max_val=self.args.task_size_max * 100)
        state = np.hstack((uav_x, uav_y, uav_z, ue_x, ue_y, ue_z, task_size, task_cpu)).astype(np.float32)
        state_n, mask_n = np.empty([self.N, self.state_dim]), np.zeros((self.N, self.M, 2))
        for n in range(self.N):
            cover_vec = cover_mat[:, n]
            state_n[n] = np.hstack((uav_x[:], uav_y[:], uav_z[:], np.where(cover_vec, ue_x, -1), np.where(cover_vec, ue_y, -1), np.where(cover_vec, ue_z, -1), np.where(cover_vec, task_size, -1), np.where(cover_vec, task_cpu, -1))).astype(np.float32)
            # 2. 离散动作的动作掩码
            mask_n[n, ~cover_vec, 1] = -np.inf
        return state, state_n, mask_n
    
    def render(self, save_dir, episode, writer=None):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectories and positions of UAVs & Users")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(self.args.uav_x_min, self.args.uav_x_max)
        ax.set_ylim(self.args.uav_y_min, self.args.uav_y_max)
        ax.set_zlim(self.args.ue_z_min, self.args.uav_z_max)
        uavs_traj_color = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        uavs_traj = np.array(self._trajectories)
        for n in range(self.N):
            ax.plot(uavs_traj[n, :, 0], uavs_traj[n, :, 1], uavs_traj[n, :, 2], 'o', color=uavs_traj_color[n], markersize=2)
        ues_color = ["#000000" if off_idx == 0 else uavs_traj_color[off_idx - 1] for off_idx in self.off_vec]
        ax.scatter(self.ues[:, 0], self.ues[:, 1], 0, marker="x", color=ues_color, s=8)
        filename = f"{episode}.png"
        plt.savefig(os.path.join(save_dir, filename))
        if writer is not None:
            writer.add_figure(filename, fig, episode)
        plt.close(fig)

    def step_action(self, act_n_dis, act_n_con):
        # 1. 保存上一时隙的无人机位置
        for n in range(self.N):
            self.trajectories[n, self.step] = self.uavs[n, :3]
        # 2. 执行 Action
        var_x = act_n_dis[:]
        var_w = act_n_con[:, :3]
        penalty_n_1, uav_pos, self.dis_mat, self.link_mat, self.nearest_link_vec = self.step_uav_movement(var_w)
        penalty_n_2, self.off_vec = self.step_offloading_decision(var_x)
        self.uavs[:, :3] = uav_pos
        # 3. 计算奖励
        reward_n = self.calc_reward()
        reward_n -= penalty_n_1
        reward_n -= penalty_n_2
        # 3. 更新到下一状态
        self.step += 1
        done = False
        if self.step == self.T:
            done = True
            state_next, obs_n_next, mask_n_next = self.reset()
        else:
            self.tasks = self.generate_tasks()
            state_next, obs_n_next, mask_n_next = self.state()
        return state_next, obs_n_next, mask_n_next, reward_n, np.repeat(done, self.N)

    def step_uav_movement(self, uav_movement_mat):
        # 1. 检查范围约束
        uav_movement_mat[:, 0] *= 2
        uav_movement_mat[:, 1] *= 2
        uav_pos = self.uavs[:, :3] + uav_movement_mat
        out_bound_x = (uav_pos[:, 0] < self.args.uav_x_min) | (uav_pos[:, 0] > self.args.uav_x_max)
        out_bound_y = (uav_pos[:, 1] < self.args.uav_y_min) | (uav_pos[:, 1] > self.args.uav_y_max)
        out_bound_z = (uav_pos[:, 2] < self.args.uav_z_min) | (uav_pos[:, 2] > self.args.uav_z_max)
        uav_pos = np.clip(uav_pos, [self.args.uav_x_min, self.args.uav_y_min, self.args.uav_z_min], [self.args.uav_x_max, self.args.uav_y_max, self.args.uav_z_max])
        penalty = np.zeros(self.N, dtype=np.float32)
        for n in range(self.N):
            if out_bound_x[n]: penalty[n] += 1
            if out_bound_y[n]: penalty[n] += 1
            if out_bound_z[n]: penalty[n] += 1
        # 2. 计算新的辅助数据结构
        ue_pos = self.ues[:, :3]
        dis_mat = np.linalg.norm(ue_pos[:, np.newaxis, :] - uav_pos, axis=-1)
        link_mat = (dis_mat <= calc_cover_radius(self.args.uav_angle, uav_pos[:, 2]).reshape(1, -1))
        nearest_link_mat = dis_mat[:]
        nearest_link_mat[~link_mat] = np.inf
        nearest_link_vec = np.argmin(nearest_link_mat, axis=-1)
        return penalty, uav_pos, dis_mat, link_mat, nearest_link_vec
    
    def step_offloading_decision(self, off_mat, dis_mat=None, link_mat=None, nearest_link_vec=None):
        # 每个用户与每个无人机之间的距离矩阵
        dis_mat = self.dis_mat if dis_mat is None else dis_mat
        # 每个用户与每个无人机是否可以链接的布尔矩阵
        link_mat = self.link_mat if link_mat is None else link_mat
        # 每个用户最近可链接的无人机向量
        nearest_link_vec = self.nearest_link_vec if nearest_link_vec is None else nearest_link_vec
        # 对于给定卸载决策，每个无人机得到的惩罚（违背卸载约束时）
        penalty = np.zeros(self.N, dtype=np.float32)
        # 计算卸载向量，并检查卸载约束
        off_vec = np.zeros(self.M, dtype=int)
        for m in range(self.M):
            if sum(off_mat[:, m]) == 0: continue
            nearest_uav = self.nearest_link_vec[m]
            for n, _ in enumerate(off_mat[:, m]):
                if nearest_uav == np.inf:
                    penalty[n] += 1
                    continue
                else:
                    off_vec[m] = nearest_uav + 1
        return penalty, off_vec

    def calc_reward(self):
        # 1. Offloading time
        t_off = np.zeros(self.M, dtype=np.float32)
        t_exe = np.zeros(self.M, dtype=np.float32)
        for m in range(self.M):
            g2a_uav = self.off_vec[m]
            # 1.1. Local
            if g2a_uav == 0:
                t_exe[m] = self.t_loc[m]
                continue
            # 1.2. Edge
            g2a_uav -= 1
            g2a_hei = self.uavs[g2a_uav, 2]
            g2a_dis = np.linalg.norm(self.ues[m, :3] - self.uavs[g2a_uav, :3])
            g2a_channel_gain = self._calc_channel_gain_g2a_or_a2g(uav_height=g2a_hei, distance=g2a_dis)
            g2a_trans_rate = self.args.g2a_bandwidth * np.log2(1 + 1 * g2a_channel_gain / self.args.g2a_noise)
            t_off[m] = self.tasks[m, 0] / g2a_trans_rate
            t_exe[m] = self.tasks[m, 1] / self.args.uav_f_max
        # 2. 计算平均优化率
        self.t_off, self.t_exe, self.t_resp = t_off, t_exe, t_off + t_exe
        opt_ratio = (self.t_loc - self.t_resp) / self.t_loc
        # opt_ratio = np.log(self.t_loc / self.t_resp)
        return np.mean(opt_ratio)
    
    def _calc_channel_gain_g2a_or_a2g(self, uav_height, distance):
        """
        基于概率信道模型，计算 G2A / A2A 信道增益
        - uav_height: 无人机高度
        - distance: 地面设备与无人机的欧氏距离
        """
        a, b, c, fr_c, eta_los, eta_nlos = self.args.env_a, self.args.env_b, self.args.env_c, self.args.env_fr_c, self.args.env_eta_los, self.args.env_eta_nlos
        power = -b * (np.degrees(np.arcsin(uav_height / distance)) - a)
        los_prob = 1 / (1 + a * (np.e ** (power)))
        nlos_prob = 1 - los_prob
        free_path_loss = 20 * np.log10(4 * np.pi / c) + 20 * np.log10(fr_c) + 20 * np.log10(distance)
        los = free_path_loss + eta_los
        nlos = free_path_loss + eta_nlos
        prob_path_loss = los_prob * los + nlos_prob * nlos
        channel_gain = 1 / prob_path_loss
        return channel_gain
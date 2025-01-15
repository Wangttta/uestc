import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import *


class UMEC:

    def __init__(self, args):
        # 1. Initialize parameters
        self.args = args
        self.M, self.N, self.T = args.m, args.n, args.t
        self.step = 0
        # 2. Initialize the UEs and UAVs
        self.ue_pos = self.__init_ue()
        self.uav_pos = self.__init_uav()
        self.ec_pos = self.__init_ec()
        self.t_loc, e_loc = None, None
        self.task, self.trajectories, self._trajectories  = self.__generate_task(), [], None
        self._ue_pos, self._uav_pos, self._task = np.copy(self.ue_pos), np.copy(self.uav_pos), np.copy(self.task)
        self.target_ue, self.target_dis, self.offloaded_ues = np.zeros(self.N, dtype=int), np.zeros(self.N), set()
        # 3. State & action dimension
        self.state_dim = 4 * self.N + 3 * self.N  # 2M tasks, 2M UEs position, 3N UAVs position
        self.action_dim = 3 * self.N     # 3N UAVs movement (continuous)

    def __init_ue(self):
        # Type 1: Random
        # x = np.random.uniform(self.args.x_min, self.args.x_max, (self.M, 1))
        # y = np.random.uniform(self.args.y_min, self.args.y_max, (self.M, 1))
        # z = np.zeros((self.M, 1))
        # return np.concatenate((x, y, z), axis=1).astype(np.float32)
        # Type 2: From file
        pre_data = pd.read_csv(self.args.dir_presets_ue).to_numpy()
        return pre_data[:self.M, -3:].astype(np.float32)

    def __init_uav(self):
        # Type 1: Random
        # x = np.random.uniform(self.args.x_min, self.args.x_max, (self.N, 1))
        # y = np.random.uniform(self.args.y_min, self.args.y_max, (self.N, 1))
        # z = np.random.uniform(self.args.z_min, self.args.z_max, (self.N, 1))
        # return np.concatenate((x, y, z), axis=1).astype(np.float32)
        # Type 2: From file
        pre_data = pd.read_csv(self.args.dir_presets_uav).to_numpy()
        return pre_data[:self.N, -3:].astype(np.float32)

    def __init_ec(self):
        # return np.zeros(3, dtype=np.float32)
        return None

    def __generate_task(self):
        """
        Each task is defined as a 2 tuple: (task_size, task_cpu_cycles)
        - task_size: total bytes in transmission (MB, acturally)
        - task_cpu_cycles: total cpu cycles required for calculating

        Returns
        ----------
        - task: ndarray (M * 2)
        """
        # 1. Random task size in bits
        task_size = np.random.randint(self.args.task_size_min, self.args.task_size_max, size=self.M).astype(np.float32)
        # 2. Total CPU cycles required
        task_cpu = (task_size * np.random.choice(self.args.task_cycle_per_bit, self.M)).astype(np.float32)
        # 3. Calculate local computing time
        self.t_loc = task_cpu * ( 2 ** 23 ) * 1e-9 / self.args.cpu_ue
        return np.concatenate((task_size.reshape(-1, 1), task_cpu.reshape(-1, 1)), axis=1).astype(np.float32)

    def state(self):
        """ 
        Return current environment state vector, contains tasks, 
        UEs & UAVs position (with min-max normalization), and 
        action mask vector. 

        Note that the mask only used to change the distribution 
        of discert actions. Continuous actions are generated 
        from Gaussian distribution and will be regenerated when 
        actions are invalid.
        """
        # 1. State info
        task_size = np.ones(self.N)
        task_cpu = np.ones(self.N)
        # task_size = min_max(self.task[:, 0], min_val=self.args.task_size_min, max_val=self.args.task_size_max)
        # task_cpu = min_max(self.task[:, 1], min_val=self.args.task_size_min * self.args.task_cycle_per_bit[0], max_val=self.args.task_size_max * self.args.task_cycle_per_bit[-1])
        # ue_x = min_max(self.ue_pos[:, 0], min_val=self.args.x_min, max_val=self.args.x_max)
        # ue_y = min_max(self.ue_pos[:, 1], min_val=self.args.y_min, max_val=self.args.y_max)

        dis_mat = np.linalg.norm(self.ue_pos[:, np.newaxis, :] - self.uav_pos[np.newaxis, :, :], axis=2)
        for n in range(self.N):
            nearest_ues = np.argsort(dis_mat[:, n])[::-1]
            offloaded_ue = -1
            for m in nearest_ues:
                if m not in self.offloaded_ues:
                    offloaded_ue = m
                    break
            offloaded_ue = 0 if offloaded_ue == -1 else offloaded_ue
            self.target_ue[n] = offloaded_ue
            self.target_dis[n] = dis_mat[offloaded_ue, n]

        ue_x = min_max(self.ue_pos[self.target_ue, 0], min_val=self.args.x_min, max_val=self.args.x_max)
        ue_y = min_max(self.ue_pos[self.target_ue, 1], min_val=self.args.y_min, max_val=self.args.y_max)
        uav_z = np.ones(self.N)
        uav_x = min_max(self.uav_pos[:, 0], min_val=self.args.x_min, max_val=self.args.x_max)
        uav_y = min_max(self.uav_pos[:, 1], min_val=self.args.y_min, max_val=self.args.y_max)
        # uav_z = min_max(self.uav_pos[:, 2], min_val=self.args.z_min, max_val=self.args.z_max)
        state = np.hstack((task_size, task_cpu, ue_x, ue_y, uav_x, uav_y, uav_z)).astype(np.float32)
        # 2. Discert action mask
        mask = np.zeros((self.N, 3, 3))
        for n in range(self.N):
            # x
            if self.uav_pos[n, 0] == self.args.x_min:
                mask[n, 0, 0] = -np.inf
            if self.uav_pos[n, 0] == self.args.x_max:
                mask[n, 0, 2] = -np.inf
            # y
            if self.uav_pos[n, 1] == self.args.y_min:
                mask[n, 1, 0] = -np.inf
            if self.uav_pos[n, 1] == self.args.y_max:
                mask[n, 1, 2] = -np.inf
            # z
            if self.uav_pos[n, 2] == self.args.z_min:
                mask[n, 2, 0] = -np.inf
            if self.uav_pos[n, 2] == self.args.z_max:
                mask[n, 2, 2] = -np.inf
        mask = mask.reshape(-1)
        return state, mask

    def reset(self):
        self.step = 0
        self._trajectories = np.copy(self.trajectories)
        self.trajectories = []
        self.target_ue, self.target_dis, self.offloaded_ues = np.zeros(self.N, dtype=int), np.zeros(self.N), set()
        self.ue_pos, self.uav_pos, self.task = np.copy(self._ue_pos), np.copy(self._uav_pos), np.copy(self._task)

    def step_action(self, action):
        """
        Execute actions in current environment

        Parameters
        ----------
        - var_w: nd_array (N * 3), UAV movement matrix, values in {0, 1}
        - var_x: nd_array (M), UEs offloading vector, values in {0, 1, ..., N}

        Returns
        ----------
        - reward: np.float32
        - next_state: nd_array
        - done: bool
        """
        # 1. Save previous trajectory
        self.trajectories.append(np.copy(self.uav_pos))
        # 2. Execute action
        self.var_x = np.ones(self.M, dtype=int)  # Offloading decisions
        self.var_w = action[:]  # UAVs movement
        # 3. Calculate reward
        reward = self.__calc_reward()
        # 4. Next state
        self.step += 1
        done = (self.step == self.args.t or len(self.offloaded_ues) == self.M)
        if done:
            self.reset()
        else:
            self.tasks = self.__generate_task()
        next_state, _ = self.state()
        return reward, next_state, done
    
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
        """
        Calculate the system cost for the current state 
        (time slot) based on all decision variables
        """
        # 1. Update UAVs positions
        uav_pos = self.uav_pos + self.var_w

        dis_mat_pre = np.linalg.norm(self.ue_pos[:, np.newaxis, :] - self.uav_pos[np.newaxis, :, :], axis=2)
        dis_mat = np.linalg.norm(self.ue_pos[:, np.newaxis, :] - uav_pos[np.newaxis, :, :], axis=2)

        reward = 0
        for n in range(self.N):
            offloading_ue = self.target_ue[n]
            distance_offset = dis_mat_pre[offloading_ue, n] - dis_mat[offloading_ue, n]
            reward += ((distance_offset + 15) / 30)
            if dis_mat[offloading_ue, n] < 200:
                self.offloaded_ues.add(offloading_ue)
                reward += 5

        self.uav_pos = uav_pos
        return reward

        # # 2. Local computing: t_loc
        # off_idx = (self.var_x != 0).astype(int)
        # t_loc = self.t_loc
        # # 3. Edge computing: t_edge
        # # 3.1. Transmission: t_off
        # a, b, c, fr_c, eta_los, eta_nlos = self.args.env_a, self.args.env_b, self.args.env_c, self.args.env_fr_c, self.args.env_eta_los, self.args.env_eta_nlos
        # target_uav_pos = self.uav_pos[self.var_x - 1, :]  # 选择本地计算的用户这里会对应最后一个无人机
        # g2a_dis = np.linalg.norm(self.ue_pos[:, :2] - target_uav_pos[:, :2], axis=-1)
        # g2a_power = -b * (np.degrees(np.arctan(target_uav_pos[:, -1] / g2a_dis)) - a)
        # g2a_los_prob = 1 / (1 + a * (np.e ** (g2a_power)))
        # g2a_nlos_prob = 1 - g2a_los_prob
        # g2a_free_path_loss = 20 * np.log10(4 * np.pi / c) + 20 * np.log10(fr_c) + 20 * np.log10(g2a_dis)
        # g2a_los = g2a_free_path_loss + eta_los
        # g2a_nlos = g2a_free_path_loss + eta_nlos
        # g2a_prob_path_loss = g2a_los_prob * g2a_los + g2a_nlos_prob * g2a_nlos
        # g2a_channel_gain = 1 / g2a_prob_path_loss
        # g2a_trans_rate = self.args.b_g2a * 1e6 * np.log2(1 + self.args.p_ue_min * g2a_channel_gain / self.args.noise_g2a) / self.M  # 单位转化 MHz => Hz
        # t_off = self.task[:, 0] * ( 2 ** 23 ) / g2a_trans_rate  # 单位转化 MByte => bit
        # # 3.2. Computing: t_exe
        # t_exe = self.task[:, 1] * ( 2 ** 23 ) * 1e-9 / self.args.cpu_uav  # 单位转化 MByte => bit, GHz => Hz
        # t_edge = t_off + t_exe
        # t_resp = (1 - off_idx) * t_loc + off_idx * t_edge
        # return -sum(t_resp / t_loc) / self.M
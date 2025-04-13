import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import *

from agent import Agent
from util import *


class ReplayBuffer:

    def __init__(self, args):
        self.T = args.t
        self.N = args.n
        self.obs_dim = args.obs_dim
        self.act_dim_dis = args.act_dim_dis[0]
        self.act_dim_con = args.act_dim_con
        self.state_dim = args.obs_dim  # 全局状态维度与每个智能体观测维度一致，只是智能体观测的状态不完整
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset()
    
    def reset(self):
        self.episode_num = 0
        self.buffer = {
            'obs_n': np.empty([self.batch_size, self.T, self.N, self.obs_dim]),
            's': np.empty([self.batch_size, self.T, self.state_dim]),
            'v_n': np.empty([self.batch_size, self.T + 1, self.N]),
            'a_n_dis': np.empty([self.batch_size, self.T, self.N, self.act_dim_dis]),
            'a_n_con': np.empty([self.batch_size, self.T, self.N, self.act_dim_con]),
            'a_n_dis_logprob': np.empty([self.batch_size, self.T, self.N, self.act_dim_dis]),
            'a_n_con_logprob': np.empty([self.batch_size, self.T, self.N, self.act_dim_con]),
            'r_n': np.empty([self.batch_size, self.T, self.N]),
            'done_n': np.empty([self.batch_size, self.T, self.N])
        }

    def store_transition(self, t, obs_n, state, value_n, action_n_dis, action_n_con, action_n_dis_logprob, action_n_con_logprob, reward_n, done_n):
        self.buffer['obs_n'][self.episode_num][t] = obs_n
        self.buffer['s'][self.episode_num][t] = state
        self.buffer['v_n'][self.episode_num][t] = value_n
        self.buffer['a_n_dis'][self.episode_num][t] = action_n_dis
        self.buffer['a_n_con'][self.episode_num][t] = action_n_con
        self.buffer['a_n_dis_logprob'][self.episode_num][t] = action_n_dis_logprob
        self.buffer['a_n_con_logprob'][self.episode_num][t] = action_n_con_logprob
        self.buffer['r_n'][self.episode_num][t] = reward_n
        self.buffer['done_n'][self.episode_num][t] = done_n

    def store_episode_value(self, t, value_n):
        self.buffer['v_n'][self.episode_num][t] = value_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        batch["a_n_dis"] = torch.tensor(self.buffer["a_n_dis"], dtype=torch.long)
        for key in self.buffer.keys():
            if key != 'a_n_dis':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch


class Critic(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.ReLU()
        orthogonal_init(self.layer1)
        orthogonal_init(self.layer2)
        orthogonal_init(self.layer3)

    def forward(self, state):
        x = self.activate_func(self.layer1(state))
        x = self.activate_func(self.layer2(x))
        value = self.layer3(x)
        return value


class MAHPPO:

    def __init__(self, args):
        # 1. 基本训练参数
        self.N, self.T = args.n, args.t
        self.n_episode = args.n_episode
        self.n_train = args.n_train
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.lr = args.lr
        self.lamda = args.lamda
        self.gamma = args.gamma
        # 2. 神经网络（分布式 Actors + 中心化 Critic）
        self.agents = [Agent(args, i) for i in range(self.N)]
        self.critic = Critic(input_dim=args.state_dim, hidden_dim=args.hidden_dim)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.buffer = ReplayBuffer(args)

    def choose_action(self, state, evaluate=False):
        with torch.no_grad():
            return self._choose_action(state, evaluate)

    def _choose_action(self, obs_n, evaluate):
        act_n_dis, act_n_con, act_n_dis_logprob, act_n_con_logprob = [], [], [], []
        for agent in self.agents:
            act_dis, act_con, act_dis_logprob, act_con_logprob = agent.choose_action(obs_n[agent.id], evaluate)
            act_n_dis.append(act_dis)
            act_n_con.append(act_con)
            act_n_dis_logprob.append(act_dis_logprob)
            act_n_con_logprob.append(act_con_logprob)
        return np.array(act_n_dis), np.array(act_n_con), np.array(act_n_dis_logprob), np.array(act_n_con_logprob)
    
    def get_value(self, state):
        with torch.no_grad():
            return self._get_value(state)

    def _get_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        value = self.critic(state)
        return value.numpy()

    def train(self, episode):
        batch = self.buffer.get_training_data()
        # 1. 计算 Generalized Advantage Estimation
        advantage, gae = [], 0
        deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
        for t in reversed(range(self.T)):
            gae = deltas[:, t] + self.gamma * self.lamda * gae
            advantage.insert(0, gae)
        advantage = torch.stack(advantage, dim=1)
        values_target = advantage + batch['v_n'][:, :-1]
        advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))
        # 2. 获取全局状态，作为 Critic Input
        critic_input = batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1)
        # 3. 按照小批次迭代训练
        actor_losses, critic_losses = [], []
        for _ in range(self.n_train):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # 3.1. 计算 Critic Loss
                values_now = self.critic(critic_input[index]).squeeze(-1)
                critic_loss = torch.mean((values_now - values_target[index]) ** 2)
                self.optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.optimizer.step()
                critic_losses.append(critic_loss.item())
                # 3.2. 依次计算每个智能体的 Actor Loss
                for agent_id, agent in enumerate(self.agents):
                    obs = batch['obs_n'][index, :, agent_id]
                    act_dis = batch['a_n_dis'][index, :, agent_id]
                    act_con = batch['a_n_con'][index, :, agent_id]
                    dis_logprob = batch['a_n_dis_logprob'][index, :, agent_id]
                    con_logprob = batch['a_n_con_logprob'][index, :, agent_id]
                    agent_advantage = advantage[index, :, agent_id]
                    actor_loss = agent.train(agent_advantage, obs, act_dis, act_con, dis_logprob, con_logprob)
                    actor_losses.append(actor_loss)
        self.lr_decay(episode)
        return actor_losses, critic_losses
    
    def lr_decay(self, episode):
        lr_now = self.lr * (1 - episode / self.n_episode)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        for agent in self.agents:
            for p in agent.optimizer.param_groups:
                p['lr'] = lr_now

    def save_model(self, save_dir, episode):
        torch.save(self.critic.state_dict(), os.path.join(save_dir, f"E{episode}_Critic.pth"))
        for agent in self.agents:
            path = os.path.join(save_dir, f"E{episode}_Agent{agent.id}.pth")
            torch.save(agent.actor.state_dict(), path)
    
    def load_model(self, load_dir):
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "Critic.pth")))
        for agent in self.agents:
            path = os.path.join(load_dir, f"Agent_{agent.id}.pth")
            agent.actor.load_state_dict(torch.load(path))

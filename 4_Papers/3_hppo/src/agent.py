import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import *

from util import *


class Actor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim_dis: tuple, output_dim_con):
        super(Actor, self).__init__()
        self.output_dim_dis = output_dim_dis
        self.output_dim_con = output_dim_con
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4_dis = nn.Linear(hidden_dim, output_dim_dis[0] * output_dim_dis[1])
        self.layer4_con_mean = nn.Linear(hidden_dim, output_dim_con)
        self.layer4_con_std = nn.Linear(hidden_dim, output_dim_con)
        self.activate_func = nn.ReLU()
        orthogonal_init(self.layer1)
        orthogonal_init(self.layer2)
        orthogonal_init(self.layer3)
        orthogonal_init(self.layer4_dis, gain=0.01)
        orthogonal_init(self.layer4_con_mean, gain=0.01)
        orthogonal_init(self.layer4_con_std, gain=0.01)

    def forward(self, obs):
        # Choose action : actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # Train         : actor_input.shape=(mini_batch_size, t, N, actor_input_dim), prob.shape(mini_batch_size, t, N, action_dim)
        x = self.activate_func(self.layer1(obs))
        x = self.activate_func(self.layer2(x))
        x = self.activate_func(self.layer3(x))
        x_dis = self.layer4_dis(x)
        x_dis = x_dis.contiguous().view(*x_dis.shape[:-1], self.output_dim_dis[0], self.output_dim_dis[1])
        dis_action_prob = torch.softmax(x_dis, dim=-1)
        con_action_mean = self.layer4_con_mean(x)
        con_action_std = self.layer4_con_std(x).clamp(-5, 2).exp()
        return dis_action_prob, con_action_mean, con_action_std


class Agent:

    def __init__(self, args, id):
        # 1. 基本训练参数
        self.id = id
        self.epsilon = args.epsilon
        self.entropy_coef = args.entropy_coef
        # 2. 神经网络
        self.actor = Actor(input_dim=args.obs_dim, hidden_dim=args.hidden_dim, output_dim_dis=args.act_dim_dis, output_dim_con=args.act_dim_con)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr, eps=1e-5)

    def choose_action(self, obs, evaluate=False):
        with torch.no_grad():
            return self._choose_action(obs, evaluate)

    def _choose_action(self, obs, evaluate):
        obs = torch.tensor(obs, dtype=torch.float32)
        dis_prob, con_mean, con_std = self.actor(obs)
        # 1. 离散动作
        if evaluate:
            act_dis = dis_prob.argmax(dim=-1)
            act_dis_logprob = None
        else:
            act_dis_dist = Categorical(probs=dis_prob)
            act_dis = act_dis_dist.sample()
            act_dis_logprob = act_dis_dist.log_prob(act_dis).numpy()
        # 2. 连续动作
        con_prob = Normal(con_mean, con_std)
        act_con = con_prob.sample()
        act_con_logprob = con_prob.log_prob(act_con).numpy()
        return act_dis.numpy(), act_con.numpy(), act_dis_logprob, act_con_logprob

    def train(self, advantage, obs, act_dis, act_con, dis_logprob, con_logprob):
        advantage = advantage[:, :, np.newaxis]
        dis_prob_now, con_mean_now, con_std_now = self.actor(obs)
        dis_dist_now = Categorical(dis_prob_now)
        con_dist_now = Normal(con_mean_now, con_std_now)
        dis_dist_entropy = dis_dist_now.entropy()
        con_dist_entropy = con_dist_now.entropy()
        dis_logprob_now = dis_dist_now.log_prob(act_dis)
        con_logprob_now = con_dist_now.log_prob(act_con)
        dis_ratio = torch.exp(dis_logprob_now - dis_logprob)
        con_ratio = torch.exp(con_logprob_now - con_logprob)
        dis_surr1 = dis_ratio * advantage
        dis_surr2 = torch.clamp(dis_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss_dis = -torch.min(dis_surr1, dis_surr2) - self.entropy_coef * dis_dist_entropy
        con_surr1 = con_ratio * advantage
        con_surr2 = torch.clamp(con_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss_con = -torch.min(con_surr1, con_surr2) - self.entropy_coef * con_dist_entropy
        actor_loss = actor_loss_dis.mean() + actor_loss_con.mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer.step()
        return actor_loss.item()

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import *

from util import *


class ReplayBuffer:

    def __init__(self, args):
        self.T = args.t
        self.N = args.n
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset()

    def reset(self):
        self.buffer = {
            'obs_n': np.empty([self.batch_size, self.T, self.N, self.obs_dim]),
            's': np.empty([self.batch_size, self.T, self.state_dim]),
            'v_n': np.empty([self.batch_size, self.T + 1, self.N]),
            'a_n': np.empty([self.batch_size, self.T, self.N]),
            'a_logprob_n': np.empty([self.batch_size, self.T, self.N]),
            'r_n': np.empty([self.batch_size, self.T, self.N]),
            'done_n': np.empty([self.batch_size, self.T, self.N])
        }
        self.episode_num = 0

    def store_transition(self, t, obs_n, state, value_n, action_n, action_logprob_n, reward_n, done_n):
        self.buffer['obs_n'][self.episode_num][t] = obs_n
        self.buffer['s'][self.episode_num][t] = state
        self.buffer['v_n'][self.episode_num][t] = value_n
        self.buffer['a_n'][self.episode_num][t] = action_n
        self.buffer['a_logprob_n'][self.episode_num][t] = action_logprob_n
        self.buffer['r_n'][self.episode_num][t] = reward_n
        self.buffer['done_n'][self.episode_num][t] = done_n

    def store_episode_value(self, t, value_n):
        self.buffer['v_n'][self.episode_num][t] = value_n
        self.episode_num += 1
    
    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activate_func = nn.ReLU()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # Choose action : actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # Train         : actor_input.shape=(mini_batch_size, t, N, actor_input_dim), prob.shape(mini_batch_size, t, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.ReLU()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # Get value : critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # Train     : critic_input.shape=(mini_batch_size, t, N, critic_input_dim), value.shape=(mini_batch_size, t, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO:

    def __init__(self, args):
        # 1. 基本训练参数
        self.T = args.t
        self.N = args.n
        self.n_episode = args.n_episode
        self.n_train = args.n_train
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.buffer = ReplayBuffer(args)
        # 2. PPO 超参数
        self.lr = args.lr
        self.lamda = args.lamda
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.entropy_coef = args.entropy_coef
        # 3. 神经网络
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        self.actor = Actor(input_dim=self.actor_input_dim, output_dim=args.action_dim, hidden_dim=args.hidden_dim)
        self.critic = Critic(input_dim=self.critic_input_dim, hidden_dim=args.hidden_dim)
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
    
    def choose_action(self, state, evaluate=False):
        with torch.no_grad():
            return self._choose_action(state, evaluate)

    def _choose_action(self, obs_n, evaluate):
        obs_n = torch.tensor(obs_n, dtype=torch.float32)
        prob = self.actor(obs_n)
        if evaluate:
            return prob.argmax(dim=-1).numpy(), None
        dist = Categorical(probs=prob)
        action_n = dist.sample()
        return action_n.numpy(), dist.log_prob(action_n).numpy()
    
    def get_value(self, state):
        with torch.no_grad():
            return self._get_value(state)
    
    def _get_value(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
        value_n = self.critic(state)  # (N, state_dim) -> (N, 1)
        return value_n.numpy().flatten()
    
    def train(self, episode):
        batch = self.buffer.get_training_data()
        advantage, gae = [], 0
        deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
        for t in reversed(range(self.T)):
            gae = deltas[:, t] + self.gamma * self.lamda * gae
            advantage.insert(0, gae)
        advantage = torch.stack(advantage, dim=1)
        values_target = advantage + batch['v_n'][:, :-1]  # v_target.shape(batch_size,t,N)
        advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))

        actor_input = batch['obs_n']
        critic_input = batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1)
        loss = [0.0]
        for _ in range(self.n_train):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                probs_now = self.actor(actor_input[index])
                values_now = self.critic(critic_input[index]).squeeze(-1)
                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                action_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                ratios = torch.exp(action_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, t, N)
                surr1 = ratios * advantage[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                critic_loss = (values_now - values_target[index]) ** 2
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
                loss.append(ac_loss.item())
        self.lr_decay(episode)
        return loss
    
    def lr_decay(self, episode):
        lr_now = self.lr * (1 - episode / self.n_episode)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Buffer:

    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], []
    
    def save(self, state, action, log_prob, reward, done, truncated=False):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done or truncated)
    
    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones = [], [], [], [], []


class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        ) 
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        state_value = self.critic(state)
        return action_prob, state_value


class PPO:

    def __init__(self, args):
        self.args = args
        self.policy = ActorCritic(args.input_dim, args.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.lr_decay_rate)
        self.buffer = Buffer()
        self.loss = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_prob, _ = self.policy(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()

    def learn(self):
        states = np.array(self.buffer.states)
        actions = np.array(self.buffer.actions)
        log_probs = np.array(self.buffer.log_probs)
        for _ in range(self.args.n_train):
            # 1. 组织当前批次的训练数据
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(log_probs)
            # 2. 计算 Advantage
            returns = self.__compute_returns(self.buffer.rewards, self.buffer.dones)
            _, action_values = self.policy(states)
            advantages = returns - action_values.detach().squeeze()
            # 3. 计算新的 LogProb
            action_probs, values = self.policy(states)
            action_dists = Categorical(action_probs)
            new_log_probs = action_dists.log_prob(actions)
            # 4. 计算 ClipRatio
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
            # 5. 计算 Loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            total_loss = actor_loss + 0.5 * critic_loss
            # 6. 执行反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.loss = total_loss.item()  # Log
        self.lr_scheduler.step()  # LR decay
        self.buffer.clear()  # 清空已经使用过的数据

    def __compute_returns(self, rewards, dones):
        returns = []
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.args.gamma * running_return
            returns.insert(0, running_return)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
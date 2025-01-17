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

    def __init__(self, input_dim, output_dim, n_agent, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(n_agent)
        ])
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.feature_net(state)
        action_probs = []
        for actor_head in self.actor_heads:
            prob = actor_head(features)
            action_probs.append(prob)
        state_value = self.critic(features)
        return action_probs, state_value


class PPO:

    def __init__(self, input_dim, output_dim, n_agent=1, n_train=10, lr=3e-4, lr_decay_rate=0.99, epsilon=0.2, gamma=0.99):
        self.n_agent, self.n_train, self.lr, self.lr_decay_rate, self.epsilon, self.gamma = n_agent, n_train, lr, lr_decay_rate, epsilon, gamma
        self.policy = ActorCritic(input_dim, output_dim, n_agent=n_agent)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=lr_decay_rate)
        self.buffer = Buffer()
        self.loss = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.policy(state)
        actions = []
        log_probs = []
        for prob in action_probs:
            dist = Categorical(prob)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
        return np.array(actions), sum(log_probs)

    def learn(self):
        states = np.array(self.buffer.states)
        actions = np.array(self.buffer.actions)
        log_probs = np.array(self.buffer.log_probs)
        for _ in range(self.n_train):
            # 1. 组织当前批次的训练数据
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(log_probs)
            # 2. 计算 Advantage
            returns = self.__compute_returns(self.buffer.rewards, self.buffer.dones)
            _, state_values = self.policy(states)
            advantages = returns - state_values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # 3. 计算新的 LogProb
            action_probs, values = self.policy(states)
            new_log_probs_list = []
            for agent_idx in range(self.n_agent):
                dist = Categorical(action_probs[agent_idx])
                agent_actions = actions[:, agent_idx]
                log_prob = dist.log_prob(agent_actions)
                new_log_probs_list.append(log_prob)
            new_log_probs = torch.stack(new_log_probs_list, dim=1).sum(dim=1)
            # 4. 计算 ClipRatio
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            # 5. 计算 Loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            total_loss = actor_loss + 0.5 * critic_loss
            # 6. 执行反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.loss = total_loss.item()  # Log
        # self.lr_scheduler.step()  # LR decay
        self.buffer.clear()  # 清空已经使用过的数据

    def __compute_returns(self, rewards, dones):
        returns = []
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns.insert(0, running_return)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
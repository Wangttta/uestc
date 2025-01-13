import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from env.umec import UMEC
from common.utils import *


class Buffer:
    
    def __init__(self, state_dim, action_dim, max_size=10000, batch_size=200, gamma=0.99, lam=0.95):
        # 1. Save parameters
        self.pointer, self.max_size, self.batch_size = 0, max_size, batch_size
        self.gamma, self.lam = gamma, lam
        # 2. Create buffers
        self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_dim[1]), dtype=np.int64)
        self.mask_buf = np.zeros((max_size, action_dim[0]), dtype=np.float32)
        self.advantage_buf = np.zeros(max_size, dtype=np.float32)
        self.reward_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.bool_)
        self.reward_t_buf = np.zeros(max_size, dtype=np.float32)
        self.value_buf = np.zeros(max_size, dtype=np.float32)
        self.log_prob_buf = np.zeros(max_size, dtype=np.float32)

    def store(self, state, action, mask, reward, done, value, log_prob):
        assert self.pointer < self.max_size
        self.state_buf[self.pointer] = state
        self.action_buf[self.pointer] = action
        self.mask_buf[self.pointer] = mask
        self.reward_buf[self.pointer] = reward
        self.done_buf[self.pointer] = done
        self.value_buf[self.pointer] = value
        self.log_prob_buf[self.pointer] = log_prob
        self.pointer += 1
        if done:
            episode_slice = slice(self.pointer - self.batch_size, self.pointer)
            rewards = np.append(self.reward_buf[episode_slice], 0)
            values = np.append(self.value_buf[episode_slice], 0)
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            self.advantage_buf[episode_slice] = discount_cumulative_sum(deltas, self.gamma * self.lam)
            self.reward_t_buf[episode_slice] = discount_cumulative_sum(rewards, self.gamma)[:-1]

    def sample(self):
        # 1. Get all training data
        state_buf = self.state_buf[:self.pointer]
        action_buf = self.action_buf[:self.pointer]
        advantage_buf = self.advantage_buf[:self.pointer]
        mask_buf = self.mask_buf[:self.pointer]
        reward_t_buf = self.reward_t_buf[:self.pointer]
        log_prob_buf = self.log_prob_buf[:self.pointer]
        # 2. Normalization trick
        state_buf = (state_buf - np.mean(state_buf)) / np.maximum(np.std(state_buf), 1e-6)
        advantage_buf = (advantage_buf - advantage_buf.mean()) / np.maximum(advantage_buf.std(), 1e-6)
        # 3. Sample (Reshuffle)
        sampler = BatchSampler(SubsetRandomSampler(range(self.pointer)), self.batch_size, drop_last=True)
        for indices in sampler:
            yield state_buf[indices], action_buf[indices], mask_buf[indices], advantage_buf[indices], reward_t_buf[indices], log_prob_buf[indices]

    def filter(self):
        state = self.state_buf[:self.pointer]
        return np.mean(state), np.std(state)

    def clear(self):
        self.pointer = 0


class DiscertActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DiscertActor, self).__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, action_dim[0])
    
    def forward(self, state, mask):
        # 1. Forward
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x_mask = self.layer4(x)
        # 2. Mask
        if mask is not None:
            x_mask += mask
        # 3. Generate discert action
        # 3.1. Probability
        # 3.1.1. Exploration phase, single input
        if state.shape[0] == self.state_dim:
            x_prob = F.softmax(x_mask.view((self.action_dim[1], -1)), dim=-1)
        # 3.1.2. Training phase, batch input
        else:
            x_prob = F.softmax(x_mask.view((x.shape[0], self.action_dim[1], -1)), dim=-1)
        # 3.2. Distribution phase
        x_dist = Categorical(x_prob)
        # 3.3. Sample
        x = x_dist.sample()
        # 3.4. Log probability
        x_log_prob = x_dist.log_prob(x).sum(len(state.shape) - 1)
        return x, x_log_prob, x_dist


class ContinuousActor(nn.Module):

    def __init__(self, state_dim, action_dim, log_std=-1.0, hidden_dim=64):
        super(ContinuousActor, self).__init__()
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.zeros(action_dim, ) + log_std, requires_grad=True)
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, mask=None):
        # 1. Forward
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x_mean = self.layer4(x)
        # 2. Generate continuous action
        # 2.1. Std
        x_std = torch.clamp(F.softplus(self.log_std), min=-1, max=1)
        # 2.2. Distribution
        x_dist = Normal(x_mean, x_std)
        x = x_dist.sample()
        # 2.3. Apply mask
        if mask is not None:
            pass
        # 2.4. Log prob
        x_log_prob = x_dist.log_prob(x).sum(len(state.shape) - 1)
        return x, x_log_prob, x_dist


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.tanh(self.layer1(state))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        value = self.layer4(x)
        return value


class PPO:

    def __init__(self, args, state_dim, action_dim, hidden_dim=64):
        """
        Parameters:
        """
        # 1. Initialize parameters
        self.args, self.state_dim, self.action_dim, self.hidden_dim = args, state_dim, action_dim, hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(state_dim, action_dim, max_size=10000, batch_size=args.n_batch)
        self.train_counter = 0
        # 2. Initialize deep networks
        self.actor = DiscertActor(state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=args.lr_decay_rate)
        self.actor_old = DiscertActor(state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_old = Critic(state_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def select_action(self, state, mask):
        # 1. Forward
        state = torch.FloatTensor(state).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        with torch.no_grad():
            state_value = self.critic_old(state)
            action, log_prob, _ = self.actor_old(state, mask)
        # 2. To CPU
        action = action.squeeze().cpu().numpy()
        log_prob = log_prob.squeeze().cpu().numpy()
        return state_value, action, log_prob
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))

    def train(self, epoch=0, writer=None):
        # 1. Initialize log info
        epoch_actor_loss, epoch_actor_loss_kl, epoch_critic_loss = 0, 0, 0
        train_counter = self.train_counter
        # 2. Training
        for i in range(self.args.n_train):
            batches = self.buffer.sample()
            for batch in batches:
                actor_loss, actor_loss_kl, critic_loss = self.__train_batch(batch)
                epoch_actor_loss += actor_loss
                epoch_actor_loss_kl += actor_loss_kl
                epoch_critic_loss += critic_loss
        # 3. Log info
        train_count = self.train_counter - train_counter
        epoch_actor_loss /= train_count
        epoch_actor_loss_kl /= train_count
        epoch_critic_loss /= epoch_critic_loss
        print(f"P2. Epoch={epoch}, train_count={train_count}, actor_loss={actor_loss}, actor_kl_loss={actor_loss_kl}, critic_loss={critic_loss}")
        writer.add_scalar('ActorLoss/Loss', epoch_actor_loss, self.train_counter)
        writer.add_scalar('ActorLoss/KL', epoch_actor_loss_kl, self.train_counter)
        writer.add_scalar('CriticLoss/Loss', epoch_critic_loss, self.train_counter)
        # 4. Update lr & network
        self.actor_lr_scheduler.step()
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def __train_batch(self, batch):
        # 1. Prepare training data
        state, action, mask, advantage, reward_t, log_prob_old = batch
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.int64, device=self.device)
        advantage = torch.as_tensor(advantage, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward_t, dtype=torch.float32, device=self.device)
        log_prob_old = torch.as_tensor(log_prob_old, dtype=torch.float32, device=self.device)
        # 2. Calculate entropy
        _, _, dist = self.actor(state, mask)
        log_prob = dist.log_prob(action.squeeze().long()).sum(1)
        dist_entropy = dist.entropy().sum(1)
        # 3. Actor loss
        ratio = torch.exp(log_prob - log_prob_old)
        clipped_advantage = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * advantage
        actor_loss = - (torch.min(ratio * advantage, clipped_advantage) + self.args.coeff_entropy * dist_entropy).mean()
        actor_loss_kl = (log_prob - log_prob_old).mean().item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), norm_type=2, max_norm=self.args.max_norm)
        self.actor_optimizer.step()
        # 4. Critic loss
        self.critic_optimizer.zero_grad()
        critic_loss = nn.SmoothL1Loss(reduction='mean')(self.critic(state).squeeze(), reward_t)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), norm_type=2, max_norm=self.args.max_norm)
        self.critic_optimizer.step()
        # 5. Training over
        self.train_counter += 1
        return actor_loss.item(), actor_loss_kl, critic_loss.item()

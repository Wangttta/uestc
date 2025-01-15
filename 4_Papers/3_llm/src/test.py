import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

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
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value

class PPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        
        self.policy = ActorCritic(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs, _ = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_returns(self, rewards, dones, values):
        returns = []
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            
            running_return = rewards[t] + self.gamma * running_return
            returns.insert(0, running_return)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.epochs):
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
            
            # Get new action probabilities and values
            probs, values = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio and surrogate losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, num_episodes=1000, max_steps=500):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            states, actions, rewards, dones = [], [], [], []
            old_log_probs = []
            
            for step in range(max_steps):
                action, log_prob = self.get_action(state)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done or truncated)
                old_log_probs.append(log_prob.item())
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
            
            # Convert states to tensor and get values
            states_tensor = torch.FloatTensor(states)
            _, values = self.policy(states_tensor)
            values = values.detach().squeeze().numpy()
            
            # Calculate returns and advantages
            returns = self.compute_returns(rewards, dones, values)
            advantages = returns - torch.FloatTensor(values)
            
            # Update policy
            self.update(states, actions, old_log_probs, returns, advantages)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}")
                self.env.render()

# Example usage
def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    ppo = PPO(env)
    ppo.train()
    env.close()

if __name__ == "__main__":
    main()
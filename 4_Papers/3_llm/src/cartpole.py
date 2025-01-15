import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algo.drl.ppo import PPO
from common.arguments import get_args

if __name__ == "__main__":

    # 1. Random parameters
    args = get_args()
    args.n_episode = 1000
    args.n_batch = 200
    args.n_train = 3

    # 2. Init environment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    env = gym.make("CartPole-v1")

    # 3. Init resolver model and run
    state_dim = env.observation_space.shape[0]
    action_dim = np.array([env.action_space.n, 1])
    ppo = PPO(args, state_dim, action_dim)

    # 4. Train
    for episode in range(args.n_episode):
        print(f"--------------------------------------------------")
        print(f"P1. Episode={episode}, collecting data ...")
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        done, episode_reward, n_step = False, 0, 0
        while not done:
            value, action, log_prob = ppo.select_action(state, mask=None)
            next_state, reward, done, truncated, info = env.step(action.item())
            next_state = torch.FloatTensor(next_state)
            ppo.buffer.store(state=state, action=action, reward=reward, mask=None, done=done, value=value, log_prob=log_prob)
            state = next_state
            episode_reward += reward
            n_step += 1
        print(f"P1. Episode={episode}, collected data, score={episode_reward}, step={n_step}")
        if ppo.buffer.pointer >= 2000:
            print(f"P2. Episode={episode}, training ...")
            ppo.train(epoch=episode, writer=None)
            ppo.buffer.clear()

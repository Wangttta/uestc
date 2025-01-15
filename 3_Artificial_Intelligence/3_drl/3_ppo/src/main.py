import gym
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ppo import PPO
from util import *

if __name__ == "__main__":

    # 1. Random parameters
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    writer = SummaryWriter(args.dir_board)

    # 2. Init environment
    env = gym.make("CartPole-v1")
    args.input_dim = env.observation_space.shape[0]  # == 4
    args.output_dim = env.action_space.n  # == 2

    # 3. Init ppo model
    ppo = PPO(args)

    # 4. Train
    for episode in range(args.n_episode):
        require_log = (episode % 10 == 0)
        # 4.1. Collect experience
        log("--------------------------------------------------", require_log)
        log(f"P1. Episode={episode}, collecting data ...", require_log)
        state, _ = env.reset()
        done, episode_reward, step = False, 0, 0
        while not done:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            ppo.buffer.save(state, action, log_prob, reward, done, truncated)
            state = next_state
            episode_reward += reward
            done = (done or truncated)
            step += 1
        writer.add_scalar('Reward', episode_reward, episode)
        log(f"P1. Episode={episode}, collect over, score={episode_reward}, step={step}", require_log)
        # 4.2. Training
        log(f"P2. Episode={episode}, training ...", require_log)
        ppo.learn()
        writer.add_scalar('Loss', ppo.loss, episode)
        log(f"P2. Episode={episode}, train over ...", require_log)

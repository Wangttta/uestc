import os
import random
import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from torch.utils.tensorboard import SummaryWriter

from mappo import MAPPO
from util import *


def collect_episode(args, env, mappo, evaluate=False):
    observations, _ = env.reset()
    obs_n = np.array([observations[agent] for agent in observations.keys()])
    episode_reward, gif_frames = 0, []
    for t in range(args.t):
        # 1. Choose action
        action_n, action_logprob_n = mappo.choose_action(obs_n, evaluate)
        state = obs_n.flatten()
        # 2. Calculate state value
        value_n = mappo.get_value(state)
        actions = {agent: action_n[i] for i, agent in enumerate(env.agents)}
        # 3. Step action
        obs_next_n, reward_n, done_n, _, _ = env.step(actions)
        done_n = np.array([done_n[agent] for agent in done_n.keys()])
        episode_reward += reward_n['agent_0']
        # 4. Save experience
        if not evaluate:
            reward_n = reward_norm(reward_n)
            mappo.buffer.store_transition(t, obs_n, state, value_n, action_n, action_logprob_n, reward_n, done_n)
            gif_frames.append(env.render())
        # 5. Next state
        obs_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])
        # 6. Done checking
        if all(done_n): 
            break
    # An episode is over, store value_n in the last step
    if not evaluate:
        state = np.array(obs_n).flatten()
        value_n = mappo.get_value(state)
        mappo.buffer.store_episode_value(t + 1, value_n)
    return episode_reward, gif_frames


if __name__ == "__main__":

    # 1. Random parameters
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 2. Create log dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logdir = args.logdir if args.logdir else ""
    runtime = timestamp()
    args.logdir = os.path.join(current_dir, logdir, runtime)
    args.logdir_model = os.path.join(args.logdir, "model")
    args.logdir_traj = os.path.join(args.logdir, "traj")
    args.logdir_board = os.path.join(args.logdir, "board")
    os.makedirs(args.logdir_model)
    os.makedirs(args.logdir_traj)
    os.makedirs(args.logdir_board)
    writer = SummaryWriter(args.logdir_board)

    # 3. Init environment
    env = simple_spread_v3.parallel_env(N=args.n, max_cycles=args.t, local_ratio=0.5, render_mode="rgb_array", continuous_actions=False)
    env.reset(seed=args.seed)
    args.obs_dim_n = [env.observation_spaces[agent].shape[0] for agent in env.agents]
    args.action_dim_n = [env.action_spaces[agent].n for agent in env.agents]
    args.obs_dim = args.obs_dim_n[0]
    args.action_dim = args.action_dim_n[0]
    args.state_dim = np.sum(args.obs_dim_n)

    # 4. Init ppo model
    mappo = MAPPO(args)
    reward_norm = Normalization(shape=args.n)

    # 5. Run
    for episode in range(args.n_episode):
        # 5.1. Eval
        if episode % args.n_eval_rate == 0:
            eval_reward = 0
            for i in range(args.n_eval):
                reward, _ = collect_episode(args, env, mappo, evaluate=True)
                eval_reward += reward
            eval_reward /= args.n_eval
            writer.add_scalar('EvalReward', eval_reward, episode)
            log(f"Episode={episode}, eval_reward={eval_reward}")
        # 5.2. Collect experience for training
        episode_reward, gif_frames = collect_episode(args, env, mappo, evaluate=False)
        writer.add_scalar('Reward', episode_reward, episode)
        # 5.3. Train
        if mappo.buffer.episode_num == args.batch_size:
            loss = mappo.train(episode)
            mappo.buffer.reset()  # Clear buffer
            writer.add_scalar('Loss', np.mean(loss), episode)
        # 5.4. Save model and trajectory
        if episode >= args.n_episode / 2 and episode % args.log_rate == 0:
            imageio.mimsave(os.path.join(args.logdir_traj, f"{episode}.gif"), gif_frames, fps=10)
            mappo.save_model(os.path.join(args.logdir_model, f"{episode}.pth"))

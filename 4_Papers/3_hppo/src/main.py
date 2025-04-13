import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from umec import UMEC
from mahppo import MAHPPO
from util import *


def collect_episode(args, env: UMEC, mappo: MAHPPO, evaluate=False):
    state, obs_n, _ = env.reset()
    episode_reward = 0
    for t in range(args.t):
        # 1. Choose action
        act_n_dis, act_n_con, act_n_dis_logprob, act_n_con_logprob = mappo.choose_action(obs_n, evaluate)
        # 2. Calculate state value
        value = mappo.get_value(state)
        # 3. Step action
        state_next, obs_n_next, reward_n, done_n = env.step_action(act_n_dis, act_n_con)
        episode_reward += np.mean(reward_n)
        # 4. Save experience
        if not evaluate:
            reward_n = reward_norm(reward_n)
            mappo.buffer.store_transition(t, obs_n, state, value, act_n_dis, act_n_con, act_n_dis_logprob, act_n_con_logprob, reward_n, done_n)
        # 5. Next state
        state, obs_n = state_next, obs_n_next
        # 6. Done checking
        if all(done_n): 
            break
    # An episode is over, store value in the last step
    if not evaluate:
        value = mappo.get_value(state)
        mappo.buffer.store_episode_value(t + 1, value)
    return episode_reward


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
    args.logdir = os.path.join(current_dir, logdir, timestamp())
    args.logdir_model = os.path.join(args.logdir, "model")
    args.logdir_traj = os.path.join(args.logdir, "traj")
    args.logdir_board = os.path.join(args.logdir, "board")
    os.makedirs(args.logdir_model)
    os.makedirs(args.logdir_traj)
    os.makedirs(args.logdir_board)
    writer = SummaryWriter(args.logdir_board)

    # 3. Init environment
    env = UMEC(args)
    env.reset()
    args.state_dim = env.state_dim
    args.obs_dim = env.obs_dim
    args.act_dim_dis = env.act_dim_dis
    args.act_dim_con = env.act_dim_con

    # 4. Init ppo model
    ppo = MAHPPO(args)
    reward_norm = Normalization(shape=args.n)

    # 5. Training & Evaluate loop
    for episode in range(args.n_episode):
        # 5.1. Evaluate
        gif_frames = None
        if episode % args.n_eval_rate == 0:
            eval_reward = 0
            for i in range(args.n_eval):
                reward = collect_episode(args, env, ppo, evaluate=True)
                eval_reward += reward
            eval_reward /= args.n_eval
            writer.add_scalar('EvalReward', eval_reward, episode)
            log(f"Evaluate Episode={episode}, avg_reward={eval_reward}")
        # 5.2. Collect experience for training
        episode_reward = collect_episode(args, env, ppo, evaluate=False)
        writer.add_scalar('Reward', episode_reward, episode)
        # 5.3. Train
        if ppo.buffer.episode_num == args.batch_size:
            actor_losses, critic_losses = ppo.train(episode)
            ppo.buffer.reset()
            writer.add_scalar('ActorLoss', np.mean(actor_losses), episode)
            writer.add_scalar('CriticLoss', np.mean(critic_losses), episode)
        # 5.4. Save model and trajectory  episode >= args.n_episode / 2 and 
        if episode % args.log_rate == 0:
            env.render(args.logdir_traj, episode)
            ppo.save_model(args.logdir_model, episode)

import argparse

"""
这里定义了当前项目运行时，所有可用的系统及命令行参数
"""
def get_args():
    parser = argparse.ArgumentParser("UMEC experiments based on LLM & DRL")
    # 1. 环境
    # 1.1. 基本设置
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    parser.add_argument("--env-name", type=str, default="UMEC", help="name of the scenario script")
    parser.add_argument("--m", type=int, default=10, help="number of users")
    parser.add_argument("--n", type=int, default=1, help="number of UAVs")
    parser.add_argument("--t", type=int, default=300, help="number of timeslots")
    parser.add_argument("--x-min", type=int, default=0, help="system area x min")
    parser.add_argument("--x-max", type=int, default=1000, help="system area x max")
    parser.add_argument("--y-min", type=int, default=0, help="system area y min")
    parser.add_argument("--y-max", type=int, default=1000, help="system area y max")
    parser.add_argument("--z-min", type=int, default=50, help="system area z min")
    parser.add_argument("--z-max", type=int, default=50, help="system area z max")
    # 1.2. 参与者属性
    parser.add_argument("--p-ue-min", type=int, default=1, help="Lower bound of transmission power for UEs (Watte)")
    parser.add_argument("--p-ue-max", type=int, default=1, help="Upper bound of transmission power for UEs (Watte)")
    parser.add_argument("--p-uav-min", type=int, default=5, help="Lower bound of transmission power for UEs (Watte)")
    parser.add_argument("--p-uav-max", type=int, default=5, help="Upper bound of transmission power for UEs (Watte)")
    parser.add_argument("--cpu-ue", type=float, default=1, help="CPU frequency of UEs (GHz)")
    parser.add_argument("--cpu-uav", type=float, default=5, help="CPU frequency of UAVs (GHz)")
    parser.add_argument("--cpu-ec", type=float, default=10, help="CPU frequency of EC (GHz)")
    parser.add_argument("--uav-angle", type=float, default=150, help="Max UAV covering angle (flare angle)")
    parser.add_argument("--max-horz-x", type=float, default=1, help="Max horizontal x movement distance (m)")
    parser.add_argument("--max-horz-y", type=float, default=1, help="Max horizontal y movement distance (m)")
    parser.add_argument("--max-vert-z", type=float, default=0, help="Max vertical z movement distance (m)")
    parser.add_argument("--max-distance", type=float, default=1.414, help="Max movement distance (m)")
    # 1.3. 通信相关参数
    parser.add_argument("--b-g2a", type=int, default=20, help="Bandwidth for ground-to-air link (MHz)")
    parser.add_argument("--b-a2a", type=int, default=40, help="Bandwidth for air-to-air link (MHz)")
    parser.add_argument("--b-a2g", type=int, default=10, help="Bandwidth for air-to-ground link (MHz)")
    parser.add_argument("--noise-g2a", type=float, default=1e-8, help="Noise of G2A link (-50 dBm, watt)")
    parser.add_argument("--noise-a2a", type=float, default=1e-13, help="Noise of A2A link (-100 dBm, watt)")
    parser.add_argument("--noise-a2g", type=float, default=1e-12, help="Noise of A2G link (-90 dBm, watt)")
    parser.add_argument("--env-a", type=float, default=4.16, help="An env constant for calculating LoS & NLoS probability")
    parser.add_argument("--env-b", type=float, default=0.72, help="An env constant for calculating LoS & NLoS probability")
    parser.add_argument("--env-c", type=float, default=3e8, help="Speed of light")
    parser.add_argument("--env-fr-c", type=float, default=1e9, help="Carrier frequency")
    parser.add_argument("--env-eta-los", type=float, default=1, help="Excessive pass loss for LoS")
    parser.add_argument("--env-eta-nlos", type=float, default=50, help="Excessive pass loss for NLoS")
    # 1.4. 任务相关参数
    parser.add_argument("--task-size-min", type=float, default=10, help="Lower bound of task size (MB)")
    parser.add_argument("--task-size-max", type=float, default=11, help="Upper bound of task size (MB)")
    parser.add_argument("--task-cycle-per-bit", type=int, nargs="*", default=[100, 100, 100], help="CPU cycles required per bit")
    parser.add_argument("--task-delay-threshold", type=float, default=1000, help="Threshold of task delay (millseconds)")
    parser.add_argument("--kappa", type=float, default=1e-28, help="Effective switched capacitance")
    parser.add_argument("--sigma", type=int, default=-100, help="Noise power (dBm)")
    # 2. DRL
    # 2.1. Training
    parser.add_argument("--n-episode", type=int, default=1000, help="Total episode for each epoch")
    parser.add_argument("--n-train", type=int, default=10, help="Training times for each episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr-decay-rate", type=float, default=0.995, help="")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip boundary")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # 2.2. Checkpoint
    parser.add_argument("--root-path", type=str, default="/Users/yangxiangdong/dev/workspace/github/uestc/4_Papers/4_ppo/data/runtime", help="root directory for saving data")
    parser.add_argument("--dir-traj", type=str, default="uav_traj", help="Directory for saving UAV trajectories")
    parser.add_argument("--dir-model", type=str, default="model", help="Directory for network model")
    parser.add_argument("--dir-board", type=str, default="tensorboard", help="Log directory of tensorboard")
    args = parser.parse_args()
    return args

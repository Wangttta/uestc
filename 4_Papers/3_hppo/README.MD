# 在自定义 UAV-MEC 环境中基于 Pytorch 的 Hybird PPO 算法实现


## 一、简介

自定义的 UAV-MEC 环境，包含多个无人机智能体，以及指定数量的终端设备（用户）。通过决策智能体的轨迹移动、任务卸载、计算资源分配等来最大化长期系统效用。


主要依赖版本：
- Python 3.8.19
- Numpy  1.24.4
- PyTorch 2.2.2
- OpenAI Gym 0.26.2

## 二、代码

```python
class UMEC()
    agents = ["agent_0", "agent_1", ...]
    observation_spaces = {
        "agent_0": gymnasium.spaces.Box(low=[0.0, 0.0, zMin], high=[xMax, yMax, zMax])
    }

// GYM API
// Box: 一个 3 * 4 维度的观测空间，每个维度的取值都是 [-1, 2]
Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)

// Box: 一个 2 维的观测空间，第一维取值 [-1, 3]，第二个维度 {-2, 4]}
Box(low=np.array([-1.0, 3.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
```

## 部分实验结果

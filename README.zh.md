语言: [English](./README.md) | 中文

# 强化学习导论

这份面向初学者的强化学习教程将帮助您全面了解强化学习的概念、理论、原理和发展历史，并教您实现一些代表性的强化学习算法，如DQN、DDPG、SAC和PPO等。

在开始学习之前，本教程建议您具备一定的概率统计理论和机器学习基础知识。具体而言，您应熟悉期望值、贝叶斯推断和马尔可夫链等概念。此外，对神经网络和激活函数的基本知识也会对您的学习过程有所帮助。如果您在这些知识点上还不太熟悉，本教程中会提供相关说明的链接，帮助您更好地理解和应用强化学习算法。

让我们开始吧！

> 本开源教程遵循[GPL-3协议](https://choosealicense.com/licenses/gpl-3.0/#)，欢迎更多的人对本仓库做出贡献。

## 目录

1. [强化学习的发展史](./tutorials/0-Overview.zh.md)
2. [强化学习中的术语和概念](./tutorials/1-Concepts.zh.md)
3. [马尔可夫过程与贝尔曼方程](./tutorials/2-MDP_and_Bellman.zh.ipynb)
4. [时序差分与Q-Learning](./tutorials/3-Temporal_difference_and_Q_learning.zh.ipynb)
5. [策略梯度方法与REINFORCE](./tutorials/4-Policy_gradient_and_REINFORCE.zh.ipynb)
6. [Actor-Critic结构与SAC](./tutorials/5-Actor_critic_and_SAC.zh.ipynb)
7. [On-policy更新模式与PPO](./tutorials/6-On_policy_and_PPO.zh.ipynb)
9. TODO: multi-agent
10. TODO: application of RL in autonomous driving

## 快速开始

### 文件夹结构

本仓库中每个文件夹的功能如下: 

```shell
.
|-- checkpoints     # 训练好的模型参数
|-- examples        # 训练、加载、测试模型的样例代码
|-- logs                # 模型训练过程中的平均奖励变化曲线
|-- rllib               # 强化学习算法的代码脚本
`-- tutorials       # 教程文档
```

### 推荐配置

TODO: 待反馈

### 训练环境配置

最简单的方式是

```shell
pip install -r requirements.txt
```

如果有conda，也可以使用

```shell
conda env create -f environment.yml
```

特别地，如果您想在Mujoco的环境中训练模型，请参考[Mujoco and Mujoco-py Installation Instructions](https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da).

### 运行`./demo_rl/`中的文件

### 查看训练记录

```shell
tensorboard --logdir ./logs --port 6006
```

## 动机

## 参考文献

以下是本教程制作过程中的所有参考资料

### 书本

[1] 强化学习导论: Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

### 论文

[1] Q-learning: Watkins, Christopher JCH, and Peter Dayan. "[Q-learning.](https://link.springer.com/article/10.1007/BF00992698)" *Machine learning* 8.3 (1992): 279-292.

[2] DQN论文初稿: Mnih, Volodymyr, et al. "[Playing atari with deep reinforcement learning.](https://arxiv.org/abs/1312.5602)" *arXiv preprint arXiv:1312.5602* (2013).

[3] DQN最终版本: Mnih, Volodymyr, et al. "[Human-level control through deep reinforcement learning.](https://www.nature.com/articles/nature14236?wm=book_wap_0005)" *nature* 518.7540 (2015): 529-533.

[4] Rainbow: Hessel, Matteo, et al. "[Rainbow: Combining improvements in deep reinforcement learning.](https://hpi.de/fileadmin/user_upload/fachgebiete/plattner/teaching/Dynamic_Pricing/Rainbow.pdf)" *Thirty-second AAAI conference on artificial intelligence*. 2018.

[5] Atari环境预处理: Machado, Marlos C., et al. "[Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents.](https://www.jair.org/index.php/jair/article/view/11182)" *Journal of Artificial Intelligence Research* 61 (2018): 523-562.

[6] DDPG: Lillicrap, Timothy P., et al. "[Continuous control with deep reinforcement learning.](https://arxiv.org/abs/1509.02971)" *arXiv preprint arXiv:1509.02971* (2015).

[7] SAC: Haarnoja, Tuomas, et al. "[Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.](https://proceedings.mlr.press/v80/haarnoja18b)" *International conference on machine learning*. PMLR, 2018.

[8] PPO: Schulman, John, et al. "[Proximal policy optimization algorithms.](https://arxiv.org/abs/1707.06347)" *arXiv preprint arXiv*:1707.06347 (2017).

### 博客与网站

[1] [Reinforcement Learning Coach](https://intellabs.github.io/coach/)

[2] [Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master)

[3] [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)



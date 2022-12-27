Language: 中文 | [English](./README.en.md)

# 强化学习导论

## 前言

在自动驾驶的决策、规划、控制领域，相比起仅由知识驱动、基于人为编写的规则判断的方法，由数据驱动、自主生成策略依据的方法（强化学习，模仿学习）具有更强的灵活性，有潜力在超出假设范围的场景中，探索到可行解。在算法选择得当的情况下，基于数据驱动的决策/规控方法将具有很强的鲁棒性和泛化性，能够运用到多样化的驾驶场景中，成为L3/L4级别自动驾驶系统的行动指示中枢。因此，在实验室内分享数据驱动类决策生成方法的知识，关于让同学们对其形成基本认知是有必要的。

## 关于

### 动机

### 特色

## 快速开始

### 文件结构

本仓库的路径结构如下：

```shell
.
│   # 在经典环境中训练原始算法的代码，和一些训练记录
├── demo_rl
│   │   # 实现的原始强化算法在不同环境下训练的average reward记录
│   ├── logs
│   │   # 训练好的模型参数
│   └── checkpoints
│   # 经典算法的原始实现
├── rllib
│   ├── algorithms
│   └── utils
│   # Jupyter Notebooks，包含了概念、原理的介绍，公式推导和算法原理，并简单展示了rllib模块如何调用
└── tutorials
```

### 环境配置

推荐系统：Ubuntu 18.04/20.04

硬件配置：

### 强化学习训练环境配置

强化学习的Mujoco环境环境需要手动配置，具体过程如下：

```shell
# 依赖下载
sudo apt-get update
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
# 针对报错 -lGL cannot be found
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

mkdir ~/.mujoco
cd ~/.mujoco
# 如果想要更新的版本请自己上https://github.com/deepmind/mujoco 下载
wget https://github.com/deepmind/mujoco/releases/download/2.3.0/mujoco-2.3.0-linux-x86_64.tar.gz
tar -zxvf mujoco-2.3.0-linux-x86_64.tar.gz -C ./mujoco230
# 需要将环境配置写到`~/.bashrc`或者其他常用的终端配置文件中
echo "export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
source ~/.bashrc
```

### Python 运行环境配置

首先需要下载miniconda包管理器。然后基于依赖表直接构建环境。

```shell
conda env create -f environment.yml
conda activate rllib
```

## 运行`./demo_rl/`中的文件

### 查看训练记录

```shell
tensorboard --logdir ./demo_rl/logs
```

## 参考资料

### 书本

[1] 强化学习导论：Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

### 论文

[1] Q-learning: Watkins, Christopher JCH, and Peter Dayan. "[Q-learning.](https://link.springer.com/article/10.1007/BF00992698)" *Machine learning* 8.3 (1992): 279-292.

[2] DQN论文初稿：Mnih, Volodymyr, et al. "[Playing atari with deep reinforcement learning.](https://arxiv.org/abs/1312.5602)" *arXiv preprint arXiv:1312.5602* (2013).

[3] DQN最终版本：Mnih, Volodymyr, et al. "[Human-level control through deep reinforcement learning.](https://www.nature.com/articles/nature14236?wm=book_wap_0005)" *nature* 518.7540 (2015): 529-533.

[4] Rainbow：Hessel, Matteo, et al. "[Rainbow: Combining improvements in deep reinforcement learning.](https://hpi.de/fileadmin/user_upload/fachgebiete/plattner/teaching/Dynamic_Pricing/Rainbow.pdf)" *Thirty-second AAAI conference on artificial intelligence*. 2018.

[5] Atari环境预处理：Machado, Marlos C., et al. "[Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents.](https://www.jair.org/index.php/jair/article/view/11182)" *Journal of Artificial Intelligence Research* 61 (2018): 523-562.

[6] DDPG：Lillicrap, Timothy P., et al. "[Continuous control with deep reinforcement learning.](https://arxiv.org/abs/1509.02971)" *arXiv preprint arXiv:1509.02971* (2015).



### 博客与网站

[Reinforcement Learning Coach](https://intellabs.github.io/coach/)


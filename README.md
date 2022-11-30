# RL101

一份面向实验室内部的RL教程，主要结构如下：

```shell
.
│   # 在经典环境中训练原始算法的代码，和一些训练记录
├── demo_rl
│   │   # 实现的原始强化算法在不同环境下训练的average reward记录
│   ├── logs
│   │   # 训练好的模型参数
│   └── models
│   # 经典算法的原始实现
├── rllib
│   ├── algorithms
│   └── utils
│   # Jupyter Notebooks，包含了概念、原理的介绍，公式推导和算法原理，并简单展示了rllib模块如何调用
└── tutorials
```

## 环境配置

推荐系统：Ubuntu 18.04/20.04

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
# 需要将环境配置写到`~/.bashrc`或者其他常用的终端配置文件中，注意修改$USER_NAME为具体用户名
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
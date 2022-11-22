import sys
sys.path.append(".")
sys.path.append("..")
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
import time
from collections import deque

import numpy as np
import gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms.ppo import PPO, PPOActor, PPOCritic


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.zeros(shape)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class StateNormalize:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x):
        self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-10)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-10)
        return x

    def reset(self):
        self.R = np.zeros(self.shape)


def train(
    env: gym.Env, agent: PPO, state_norm, reward_scale,
    n_step: int, env_name: str, writer: SummaryWriter):

    done = False
    score = 0
    scores = []
    average_return = 0
    state = env.reset()
    cnt_episode = 0
    
    if state_norm:
        state_normalize = StateNormalize(env.observation_space.shape)
    if reward_scale:
        reward_scaler = RewardScaling(1, 0.99)

    for i in range(n_step):
        if done:
            print("Episode: %d, score: %f, buffer capacity: %d" \
                % (cnt_episode, score, len(agent.buffer)))
            cnt_episode +=1
            if reward_scale:
                reward_scaler.reset()
            scores.append(score)
            score = 0
            state = env.reset()
            done = False

        # env.render() # render mujoco environment will take a large part of cpu resource, though it's funny.
        if state_norm:
            state = state_normalize(state)
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        if state_norm:
            state = state_normalize(next_state)
        score += reward
        if reward_scale:
            reward = reward_scaler(reward)[0]
        done_mask = 0.0 if done else 1.0
        agent.buffer.push((state, action, reward, done_mask, next_state, log_prob))
        state = next_state

        agent.train()

        if i % 1000 == 0:
            if len(scores) > 0:
                average_return = np.mean(scores)
                scores.clear()
            print("The average return is %f" % average_return)
            writer.add_scalar("%s_average_return" % env_name, average_return, i)
        
    env.close()


def tensorboard_writer(env_name):
    """Generate a tensorboard writer
    """
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    writer_path = "./logs/ppo/max/%s/%s/" % (env_name, timestamp)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


def PPO_min_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(1e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
    }
    # Generate agent
    agent = PPO(configs)
    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, False, True, n_step, env_name, writer)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PPOActorInit(PPOActor):
    def __init__(
        self, discrete: bool, state_dim: int, action_dim: int, hidden_size: int
    ):
        super().__init__(discrete, state_dim, action_dim, hidden_size)

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)
    
    def forward(self, state: torch.Tensor):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        return mean


class PPOCriticInit(PPOCritic):
    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__(state_dim, hidden_size)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, state: torch.Tensor):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def PPO_max_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(1e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "actor_net": PPOActorInit,
        "critic_net": PPOCriticInit,
        "advantage_norm": True,
        "entropy_coef": 0.01,
        "gradient_clip": True
    }
    # Generate agent
    agent = PPO(configs)
    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, False, False, n_step, env_name, writer)


def PPO_atari(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(1e6)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2" # n_step=int(3e6)
    # env_name = "Ant-v2" # n_step=int(3e6)
    # env_name = "Humanoid-v3" # n_step = int(1e7)
    # PPO_min_mujoco(env_name)
    PPO_max_mujoco(env_name)
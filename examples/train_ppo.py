import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../rllib")

import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms import ppo

device = torch.device("cuda:2")

class CheckpointWriter:
    def __init__(self, path: str, prefix: str, num_ckpt: int = 10):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.prefix = prefix
        self.num_ckpt = num_ckpt
        self.ckpt_idx = []

    def save(self, agent, average_return: int, suffix: int):
        self.ckpt_idx.append((average_return, suffix))
        self.ckpt_idx.sort(key=lambda idx: idx[0], reverse=True)

        if len(self.ckpt_idx) <= self.num_ckpt:
            path = "%s/%s_%d.pt" % (self.path, self.prefix, suffix)
            agent.save(path)
        else:
            remove_idx = self.ckpt_idx.pop()
            if remove_idx[1] != suffix:
                save_path = "%s/%s_%d.pt" % (self.path, self.prefix, suffix)
                agent.save(save_path)
                remove_path = "%s/%s_%d.pt" % (self.path, self.prefix, remove_idx[1])
                os.remove(remove_path)



class PPOAtariActor(nn.Module):
    def __init__(self, num_channels: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        x = self.net(state)
        return x

    def get_dist(self, state: torch.Tensor) -> torch.distributions.Categorical:
        x = self.forward(state)
        dist = torch.distributions.Categorical(x)

        return dist

    def action(self, state: torch.Tensor) -> int:
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy


class PPOAtariCritic(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        x = self.net(state)
        return x


def make_atari_env(id: str):
    env_config = {
        "id": id + "NoFrameskip-v4",
        "obs_type": "rgb",
        "full_action_space": True,
        "render_mode": "rgb_array",
    }
    env = gym.make(**env_config)
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


if __name__ == "__main__":
    env_name = "CartPole-v1"
    # env_name = "Acrobot-v1"
    num_envs = 3
    num_iterations = 100000

    envs = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
    states, info = envs.reset()
    scores = deque(maxlen=10)
    score_cache = [0] * num_envs

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)

    agent_configs_classic_control = ppo.PPOConfig(
        {
            "state_space": envs.single_observation_space,
            "action_space": envs.single_action_space,
            "continuous": False,
            "num_envs": num_envs,
            "num_epochs": 4,
            "horizon": 64,
            "batch_size": 32,
            "clip_epsilon": 0.1,
            "norm_advantage": False,
            "clip_grad_norm": False,
            "vf_coef": 0.1,
        }
    )

    agent = ppo.PPO(agent_configs_classic_control, device)

    log_tag = f"ClassicControl/{env_name}"
    log_writer = SummaryWriter(f"../logs/{agent.name}/{log_tag}/{timestamp}")
    ckpt_writer = CheckpointWriter(
        f"../models/{agent.name}/{log_tag}/{timestamp}", agent.name, 10
    )

    for step in range(num_iterations):
        actions, log_probs, values = agent.get_action(states)
        observations = envs.step(actions)
        transitions = (observations, states, actions, log_probs, values)
        agent.push(transitions)

        next_states, rewards, terminates, truncates, _ = observations
        states = next_states

        agent.train()

        for i in range(num_envs):
            score_cache[i] += rewards[i]
            if terminates[i] or truncates[i]:
                scores.append(score_cache[i])
                score_cache[i] = 0

        if step > 0 and step % 200 == 0 and len(scores) > 0:
            average_return = np.mean(scores)
            log_writer.add_scalar(f"ClassicControl/{env_name}", average_return, step * 3)
            ckpt_writer.save(agent, average_return, step * 3)
            print(f"Step: {step}, Score: {average_return}")
            

    envs.close()

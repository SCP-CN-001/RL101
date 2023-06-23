#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: networks.py
# @Description: Some networks that may be used for RL training.
# @Time: 2023/06/19
# @Author: Yueyuan Li

import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../rllib")

import torch
import torch.nn as nn

from rllib.algorithms import dqn, rainbow


class LinearQNetwork(dqn.QNetwork):
    """The Q-network for classic control tasks."""

    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128):
        """Initialization."""
        super().__init__(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )


class LinearRainbowQNetwork(rainbow.RainbowQNetwork):
    """The Q-network for classic control tasks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        support: torch.Tensor,
        hidden_size: int = 128,
        n_atoms: int = 51,
    ):
        """Initialization."""
        super().__init__(in_dim, out_dim, n_atoms, support)

        self.feature = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU())

        self.advantage_hidden = rainbow.NoisyLinear(hidden_size, hidden_size)
        self.advantage = rainbow.NoisyLinear(hidden_size, out_dim * n_atoms)

        self.value_hidden = rainbow.NoisyLinear(hidden_size, hidden_size)
        self.value = rainbow.NoisyLinear(hidden_size, n_atoms)


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

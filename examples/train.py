#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: train.py
# @Description:
# @Time: 2023/05/28
# @Author: Yueyuan Li

import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../rllib")

import argparse
import json

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from examples.trainer import trainer
from rllib.algorithms import dqn, rainbow, ddpg, td3, sac, ppo


class LinearQNetwork(dqn.QNetwork):
    """The Q-network for classic control tasks."""

    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super().__init__(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )


class LinearRainbowQNetwork(rainbow.RainbowQNetwork):
    """The Q-network for classic control tasks."""

    def __init__(self, in_dim: int, out_dim: int, n_atoms: int, support: torch.Tensor):
        """Initialization."""
        super().__init__(in_dim, out_dim, n_atoms, support)

        self.feature = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())

        self.advantage_hidden = rainbow.NoisyLinear(128, 128)
        self.advantage = rainbow.NoisyLinear(128, out_dim * n_atoms)

        self.value_hidden = rainbow.NoisyLinear(128, 128)
        self.value = rainbow.NoisyLinear(128, n_atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--max-step", type=int, default=int(1e6))
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--score-length", type=int, default=100)
    parser.add_argument("--record-log", action="store_true", default=False)
    parser.add_argument("--record-interval", type=int, default=1000)
    parser.add_argument("--record-ckpt", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open("train.config", "r") as f:
        config = json.load(f)

    if args.tag == "Atari":
        env_config = {
            "id": args.env + "NoFrameskip-v4",
            "obs_type": "rgb",
            "full_action_space": True,
            "render_mode": "rgb_array",
        }
        env = gym.make(**env_config)
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
        env = gym.wrappers.FrameStack(env, num_stack=4)

    elif args.tag == "ClassicControl":
        env = gym.make(args.env)

    elif args.tag == "Mujoco":
        env = gym.make(args.env)
        if args.agent == "ppo":
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    else:
        raise NotImplementedError(f"Environment not found: {args.tag}")

    agent_config_dict = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        **config[args.tag]["common"],
        **(config[args.tag][args.agent] if args.agent in config[args.tag] else {}),
    }

    # make agent
    device = torch.device(args.device)

    if args.agent == "dqn":
        if args.tag == "ClassicControl":
            agent_config_dict["q_net"] = LinearQNetwork
            agent_config_dict["q_net_kwargs"] = {
                "in_dim": env.observation_space.shape[0],
                "out_dim": env.action_space.n,
            }

        agent_config = dqn.DQNConfig(agent_config_dict)
        agent = dqn.DQN(agent_config, device=device)

    elif args.agent == "rainbow":
        if args.tag == "ClassicControl":
            agent_config_dict["q_net"] = LinearRainbowQNetwork
            agent_config_dict["q_net_kwargs"] = {
                "in_dim": env.observation_space.shape[0],
                "out_dim": env.action_space.n,
                "n_atoms": 51,
            }

        agent_config = rainbow.RainbowConfig(agent_config_dict)
        agent = rainbow.Rainbow(agent_config, device=device)

    elif args.agent == "ddpg":
        agent_config = ddpg.DDPGConfig(agent_config_dict)
        agent = ddpg.DDPG(agent_config, device=device)

    elif args.agent == "td3":
        agent_config = td3.TD3Config(agent_config_dict)
        agent = td3.TD3(agent_config, device=device)

    elif args.agent == "sac":
        agent_config = sac.SACConfig(agent_config_dict)
        agent = sac.SAC(agent_config, device=device)

    elif args.agent == "ppo":
        if args.tag == "ClassicControl":
            if args.env in ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]:
                agent_config_dict["continuous"] = False

        agent_config = ppo.PPOConfig(agent_config_dict)
        agent = ppo.PPO(agent_config, device=device)

    trainer(
        env,
        agent,
        max_step=args.max_step,
        debug=args.debug,
        score_length=args.score_length,
        record_log=args.record_log,
        log_tag=f"{args.tag}/{args.env}",
        record_ckpt=args.record_ckpt,
        record_interval=args.record_interval,
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: trainer.py
# @Description:
# @Time: 2023/05/22
# @Author: Yueyuan Li

import sys

sys.path.append(".")

import time
import argparse
import json
from collections import deque

import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from networks import LinearQNetwork, LinearRainbowQNetwork
from networks import PPOAtariActor, PPOAtariCritic
from ckpt_writer import CheckpointWriter

from rllib.algorithms import dqn, rainbow, ddpg, sac, td3, ppo


def make_atari_env(env_id: str) -> gym.Env:
    env_config = {
        "id": env_id + "NoFrameskip-v4",
        "obs_type": "rgb",
        "full_action_space": True,
        "render_mode": "rgb_array",
    }

    env = gym.make(**env_config)
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


def ppo_wrapper(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    return env


def trainer(
    envs,                   # VectorEnv
    agent: callable,    # AgentBase
    max_step: int,
    debug: bool = False,
    score_length: int = 100,
    record_log: bool = True,
    log_tag: str = "",
    record_ckpt: bool = True,
    record_interval: int = 1000,
):
    scores = deque([], maxlen=score_length)
    score_cache = [0] * envs.num_envs
    if debug:
        losses = deque([], maxlen=score_length)
    step_cnt = 0
    episode_cnt = 0

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)

    if record_log:
        log_writer = SummaryWriter(log_dir=f"../logs/{agent.name}/{log_tag}/{timestamp}")

    if record_ckpt:
        ckpt_writer = CheckpointWriter(
            f"../models/{agent.name}/{log_tag}/{timestamp}", agent.name, 10
        )

    while step_cnt < max_step:
        states, _ = envs.reset()
        done = False

        while not done:
            if agent.name == "PPO":
                actions, log_probs, values = agent.get_action(np.array(states))
                next_states, rewards, terminated, truncated, info = envs.step(actions)
                reward_mask = np.sign(rewards) if "Atari" in log_tag else rewards
                transition = ((next_states, rewards, terminated, truncated, info), states, actions, log_probs, values)
            else:
                action = agent.get_action(np.array(states[0]))
                next_states, rewards, terminated, truncated, info = envs.step([action])
                reward_mask = np.sign(rewards[0]) if "Atari" in log_tag else rewards[0]
                done = int(terminated[0] or truncated[0])
                transition = (states[0], action, next_states[0], reward_mask, done)

            agent.push(transition)

            if debug:
                loss = agent.train()
            else:
                agent.train()
            states = next_states

            for i in range(envs.num_envs):
                score_cache[i] += rewards[i]
                if terminated[i] or truncated[i]:
                    scores.append(score_cache[i])
                    score_cache[i] = 0

            step_cnt += 1
            if debug and loss is not None:
                losses.append(loss.cpu().detach().numpy())

            if len(scores) > 0 and step_cnt % record_interval == 0:
                average_return = np.mean(scores)
                print(
                    "Current Episode: %d, Total Step: %d, Average Return: %.2f"
                    % (episode_cnt, step_cnt, average_return)
                )

                if record_log:
                    log_writer.add_scalar(log_tag, average_return, step_cnt)
                    if debug and len(losses) > 0:
                        average_loss = np.mean(losses)
                        log_writer.add_scalar(log_tag + "/loss", average_loss, step_cnt)

                if record_ckpt:
                    ckpt_writer.save(agent, int(average_return), step_cnt)

        episode_cnt += 1

    envs.close()


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
    parser.add_argument("--continue-training", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open("train.config", "r") as f:
        config = json.load(f)

    # create async environments
    try:
            num_envs = config[args.tag][args.agent]["num_envs"]
    except:
        num_envs = 1

    if args.tag == "Atari":
        envs = gym.vector.AsyncVectorEnv(
            [lambda: make_atari_env(args.env)] * num_envs
        )
    elif args.tag == "ClassicControl":
        envs = gym.vector.make(args.env, num_envs=num_envs)
    elif args.tag == "Mujoco":
        if args.agent == "ppo":
            envs = gym.vector.AsyncVectorEnv(
                [lambda: ppo_wrapper(args.env)] * num_envs
            )
        else:
            envs = gym.vector.make(args.env, num_envs=num_envs)
    else:
        raise NotImplementedError(f"Environment not found: {args.tag}")

    agent_config_dict = {
        "state_space": envs.single_observation_space,
        "action_space": envs.single_action_space,
        **config[args.tag]["common"],
        **(config[args.tag][args.agent] if args.agent in config[args.tag] else {}),
    }

    # make agent
    device = torch.device(args.device)

    if args.agent == "dqn":
        if args.tag == "ClassicControl":
            agent_config_dict = {
                "q_net": LinearQNetwork,
                "q_net_kwargs": {
                    "in_dim": agent_config_dict["state_space"].shape[0],
                    "out_dim": agent_config_dict["action_space"].n,
                },
                **agent_config_dict,
            }

        agent_config = dqn.DQNConfig(agent_config_dict)
        agent = dqn.DQN(agent_config, device=device)

    elif args.agent == "rainbow":
        if args.tag == "ClassicControl":
            agent_config_dict = {
                "q_net": LinearRainbowQNetwork,
                "q_net_kwargs": {
                    "in_dim": agent_config_dict["state_space"].shape[0],
                    "out_dim": agent_config_dict["action_space"].n,
                    "n_atoms": 51,
                },
                **agent_config_dict,
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
            if "Pendulum" not in args.env and "MountainCarContinuous" not in args.env:
                agent_config_dict = {**agent_config_dict, **config[args.tag]["ppo_discrete"]}
        elif args.tag == "Atari":
            agent_config_dict = {
                "actor_net": PPOAtariActor,
                "actor_kwargs": {
                    "num_channels": agent_config_dict["state_space"].shape[0],
                    "action_dim": agent_config_dict["action_space"].n,
                },
                "critic_net": PPOAtariCritic,
                "critic_kwargs": {
                    "num_channels": agent_config_dict["state_space"].shape[0],
                },
                **agent_config_dict
            }

        agent_config = ppo.PPOConfig(agent_config_dict)
        agent = ppo.PPO(agent_config, device=device)

    trainer(
        envs,
        agent,
        max_step=args.max_step,
        debug=args.debug,
        score_length=args.score_length,
        record_log=args.record_log,
        log_tag=f"{args.tag}/{args.env}",
        record_ckpt=args.record_ckpt,
        record_interval=args.record_interval,
    )

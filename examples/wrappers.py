#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: wrappers.py
# @Description: This file is used to wrap the environment.
# @Time: 2023/06/26
# @Author: Yueyuan Li

import gymnasium as gym


def atari_wrapper(env_id: str) -> gym.Env:
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


def mujoco_ppo_wrapper(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    return env

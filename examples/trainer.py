#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: trainer.py
# @Description:
# @Time: 2023/05/22
# @Author: Yueyuan Li

import sys

sys.path.append(".")

import time
from collections import deque

# import objgraph
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from .ckpt_writer import CheckpointWriter


def trainer(
    env: gym.Env,
    agent: callable,  # AgentBase
    max_step: int,
    record_log: bool = True,
    log_tag: str = "",
    record_ckpt: bool = True,
    record_interval: int = 1000,
):
    scores = deque([], maxlen=100)
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
        score = 0
        state, _ = env.reset()
        done = False

        while not done:
            if agent.name == "PPO":
                action, log_prob, value = agent.get_action(state)
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = int(terminated or truncated)

            reward_mask = np.sign(reward) if "Atari" in log_tag else reward

            if hasattr(agent, "buffer"):
                if agent.name == "PPO":
                    transition = (state, action, reward_mask, done, log_prob, value)
                else:
                    transition = (state, action, reward_mask, done)

                agent.buffer.push(transition, next_state)

            state = next_state
            score += reward
            agent.train()

            step_cnt += 1

            if len(scores) > 0 and step_cnt % record_interval == 0:
                average_return = np.mean(scores)
                print(
                    "Current Episode: %d, Total Step: %d, Average Return: %.2f"
                    % (episode_cnt, step_cnt, average_return)
                )

                if record_log:
                    log_writer.add_scalar(log_tag, average_return, step_cnt)

                if record_ckpt:
                    ckpt_writer.save(agent, int(average_return), step_cnt)

        scores.append(score)
        episode_cnt += 1
        # objgraph.show_growth()

    env.close()

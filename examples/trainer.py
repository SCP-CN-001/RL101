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

import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from .ckpt_writer import CheckpointWriter


def trainer(
    env: gym.Env,
    agent: callable,  # AgentBase
    max_step: int,
    debug: bool = False,
    score_length: int = 100,
    record_log: bool = True,
    log_tag: str = "",
    record_ckpt: bool = True,
    record_interval: int = 1000,
):
    scores = deque([], maxlen=score_length)
    # losses = deque([], maxlen=score_length)
    if debug:
        act_time = deque([], maxlen=score_length)
        simulate_time = deque([], maxlen=score_length)
        memorize_time = deque([], maxlen=score_length)
        train_time = deque([], maxlen=score_length)
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
            t1 = time.time()
            if agent.name == "PPO":
                action, log_prob, value = agent.get_action(np.array(state))
            else:
                action = agent.get_action(np.array(state))

            t2 = time.time()
            next_state, reward, terminated, truncated, _ = env.step(action)

            reward = np.sign(reward) if "Atari" in log_tag else float(reward)
            done = int(terminated or truncated)

            t3 = time.time()
            if hasattr(agent, "buffer"):
                if agent.name == "PPO":
                    transition = (state, action, next_state, reward, done, log_prob, value.item())
                else:
                    transition = (state, action, next_state, reward, done)

                agent.push(transition)

            t4 = time.time()
            agent.train()
            # loss = agent.train()
            state = next_state
            score += reward

            step_cnt += 1
            # if loss is not None:
            #     losses.append(loss.cpu().detach().numpy())

            if len(scores) > 0 and step_cnt % record_interval == 0:
                average_return = np.mean(scores)
                print(
                    "Current Episode: %d, Total Step: %d, Average Return: %.2f"
                    % (episode_cnt, step_cnt, average_return)
                )

                if record_log:
                    log_writer.add_scalar(log_tag, average_return, step_cnt)
                    # if len(losses) > 0:
                    #     average_loss = np.mean(losses)
                    #     log_writer.add_scalar(log_tag + "/loss", average_loss, step_cnt)

                if debug:
                    act_time.append(t2 - t1)
                    simulate_time.append(t3 - t2)
                    memorize_time.append(t4 - t3)
                    train_time.append(time.time() - t4)
                    print("Average Action Time: %.4f" % np.mean(act_time))
                    print("Average Simulate Time: %.4f" % np.mean(simulate_time))
                    print("Average Memorize Time: %.4f" % np.mean(memorize_time))
                    print("Average Train Time: %.4f" % np.mean(train_time))

                if record_ckpt:
                    ckpt_writer.save(agent, int(average_return), step_cnt)

        scores.append(score)
        episode_cnt += 1
        # objgraph.show_growth()

    env.close()

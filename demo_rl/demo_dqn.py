import sys
sys.path.append(".")
sys.path.append("..")
import os
import time
from collections import deque

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms.dqn import DQN
from rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from rllib.utils.env_wrapper.atari_wrapper import AtariPreprocessing


np.random.seed(20)


def tensorboard_writer(env_name):
    """Generate a tensorboard writer
    """
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    writer_path = "./logs/dqn/%s/%s/" % (env_name, timestamp)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


def train(
    env: gym.Env, agent:DQN, num_epoch: int,
    env_name: str, writer: SummaryWriter
):

    scores = deque([], maxlen=100)
    name = env_name.split("/")[-1]
    epoch = 0

    while epoch < num_epoch:
        score = 0
        state = env.reset()
        done = False
        
        while not done:
            state = np.array(state)
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, done_mask))
            state = next_state
            score += reward

            agent.train()
            if agent.update_cnt % 50000 == 1:
                epoch = agent.update_cnt // 50000
                writer.add_scalar("%s_average_return" % name, np.mean(scores), epoch)
                print(
                    "epoch: %d, average_return: %f, buffer_capacity: %d" % \
                        (epoch, np.mean(scores), len(agent.buffer))
                )
        
        scores.append(score)

    env.close()


def DQN_atari(env_name):
    # Generate environment
    env = gym.make(env_name, render_mode='rgb_array')
    env = AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    # Params
    num_epoch = 1000
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "target_update_freq": 1e3,
        "batch_size": 32,
        "replay_start_size": 5e4,
        "buffer_size": int(1e5),
        "gradient_clip": True
    }

    # Generate agent
    agent = DQN(configs)
    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, num_epoch, env_name, writer)


if __name__ == '__main__':
    # much better than human's performance
    # DQN_atari("VideoPinballNoFrameskip-v4")
    DQN_atari("PongNoFrameskip-v4")
    #   use as a example experiment in the paper
    # DQN_atari("SpaceInvadersNoFrameskip-v4")
    # close to human's performance
    # DQN_atari("FreewayNoFrameskip-v4")
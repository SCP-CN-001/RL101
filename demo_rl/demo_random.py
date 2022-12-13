import sys
sys.path.append(".")
sys.path.append("..")
import time

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

from rllib.utils.env_wrapper.atari_wrapper import AtariPreprocessing
from demo_rl.trainer import mujoco_trainer
from demo_rl.utils import writer_generator


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, state):
        return self.action_space.sample()
    
    def train(self):
        return


def random_mujoco(env_name):
    env = gym.make(env_name)
    agent = RandomAgent(env.action_space)
    log_writer, _ = writer_generator("random", env_name, "step")
    n_step = int(1e6)
    mujoco_trainer(env, agent, n_step, env_name, log_writer)


def random_atari():
    agent = RandomAgent()


if __name__ == "__main__":
    # mujoco
    # env_name = "Hopper-v3"
    env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2" # n_step=int(3e6)
    # env_name = "Ant-v2" # n_step=int(3e6)
    random_mujoco(env_name)

    # atari


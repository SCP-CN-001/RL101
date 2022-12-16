import sys
sys.path.append(".")
sys.path.append("..")

import gym

from rllib.algorithms.ddpg import DDPG
from demo_rl.trainer import mujoco_trainer
from demo_rl.utils import writer_generator


def DDPG_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(3e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
    }
    # Generate agent
    agent = DDPG(configs)
    # Record training process
    log_writer, ckpt_writer = writer_generator("ddpg", env_name, "step")
    mujoco_trainer(env, agent, n_step, log_writer=log_writer, ckpt_writer=ckpt_writer)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2"
    # env_name = "Ant-v2"
    DDPG_mujoco(env_name)
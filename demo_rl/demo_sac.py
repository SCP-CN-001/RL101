import sys
sys.path.append(".")
sys.path.append("..")

import gym

from rllib.algorithms.sac import SAC
from demo_rl.mujoco_trainer import mujoco_trainer
from demo_rl.utils import writer_generator


def SAC_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(1e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space
    }
    # Generate agent
    agent = SAC(configs)
    # Record training process
    log_writer, ckpt_writer = writer_generator("sac", env_name, "step")
    mujoco_trainer(env, agent, n_step, env_name, log_writer, ckpt_writer)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2" # n_step=int(3e6)
    # env_name = "Ant-v2" # n_step=int(3e6)
    SAC_mujoco(env_name)
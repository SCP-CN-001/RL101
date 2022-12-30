import sys
sys.path.append(".")
sys.path.append("..")

import gym
from torch.utils.tensorboard import SummaryWriter

from rllib.utils.env_wrapper.atari_wrapper import AtariPreprocessing
from demo_rl.trainer import mujoco_trainer, atari_trainer
from demo_rl.utils import writer_generator


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, _):
        return self.action_space.sample()
    
    def train(self):
        return


def random_mujoco(env_name, n_step):
    env = gym.make(env_name)
    agent = RandomAgent(env.action_space)
    log_writer, _ = writer_generator("random", env_name, "step")
    mujoco_trainer(env, agent, n_step=n_step, log_writer=log_writer)


def random_atari(env_name, n_step):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    agent = RandomAgent(env.action_space)
    log_writer, _ = writer_generator("random", env_name, "step")
    atari_trainer(env, agent, n_step=n_step, log_writer=log_writer)


if __name__ == "__main__":
    # mujoco
    # random_mujoco("Hopper-v3", int(3e6))
    # random_mujoco("Walker2d-v3", int(3e6))
    # random_mujoco("HalfCheetah-v2", int(3e6))
    # random_mujoco("Ant-v2", int(3e6))

    # atari
    # random_atari("BreakoutNoFrameskip-v4", int(5e7))
    # random_atari("FreewayNoFrameskip-v4", int(5e7))
    random_atari("PongNoFrameskip-v4", int(5e7))
    # random_atari("SeaquestNoFrameskip-v4", int(5e7))
    # random_atari("SpaceInvadersNoFrameskip-v4", int(5e7))
    # random_atari("EnduroNoFrameskip-v4", int(1e7))
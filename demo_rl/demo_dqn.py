import sys
sys.path.append(".")
sys.path.append("..")

import gym

from rllib.algorithms.dqn import DQN
from rllib.utils.env_wrapper.atari_wrapper import AtariPreprocessing
from demo_rl.trainer import atari_trainer
from demo_rl.utils import writer_generator


def DQN_atari(env_name):
    # Generate environment
    env = gym.make(env_name, render_mode='rgb_array')
    env = AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    # Params
    n_step = int(5e7)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "target_update_freq": 1e3,
        "replay_start_size": 5e4,
        "buffer_size": int(1e5),
        "gradient_clip": True
    }
    
    # Generate agent
    agent = DQN(configs)
    # Generate tensorboard writer
    log_writer, ckpt_writer = writer_generator("dqn", env_name, "step")
    atari_trainer(env, agent, n_step=n_step, log_writer=log_writer, ckpt_writer=ckpt_writer)


if __name__ == '__main__':
    env_name = "BreakoutNoFrameskip-v4"
    # env_name = "FreewayNoFrameskip-v4"
    # env_name = "PongNoFrameskip-v4"
    # env_name = "SeaquestNoFrameskip-v4"
    # env_name = "SpaceInvadersNoFrameskip-v4"

    DQN_atari(env_name)
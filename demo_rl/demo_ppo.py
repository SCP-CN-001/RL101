import sys
sys.path.append(".")
sys.path.append("..")
from collections import deque

import numpy as np
import gym

from rllib.algorithms.base.agent import AgentBase
from rllib.algorithms.ppo import PPO
from demo_rl.utils import ActionWrapper
from demo_rl.trainer import mujoco_trainer
from demo_rl.utils import writer_generator


def mujoco_trainer(
    env: gym.Env, agent: AgentBase, 
    n_step: int = int(1e6), record_interval: int = 5000, print_interval: int = 10,
    log_writer = None, ckpt_writer = None, restart = False, restart_step = None
):
    env_name = env.unwrapped.spec.id
    action_processor = ActionWrapper(env.action_space)

    scores = deque([], maxlen=10)
    step_cnt = 0
    episode_cnt = 0

    if restart:
        step_cnt = restart_step

    while step_cnt < n_step:
        score = 0
        state = env.reset()
        done = False

        while not done:
            # env.render() # render mujoco environment will take a large part of cpu resource, though it's funny.
            action, log_std = agent.get_action(state)
            # action output range[-1,1], expand to allowable range
            action_in =  action_processor.process(action)
            next_state, reward, done, _ = env.step(action_in)
            done_mask = 0.0 if done else 1.0
            if hasattr(agent, "buffer"):
                agent.buffer.push((state, action, reward, done_mask, log_std))
            state = next_state
            score += reward
            agent.train()
            
            step_cnt += 1
            if step_cnt % record_interval == 0 and log_writer is not None:
                if len(scores) == 0:
                    continue
                average_return = np.mean(scores)
                log_writer.record_scalar("%s_average_return" % env_name, average_return, step_cnt)
                if ckpt_writer is not None:
                    ckpt_writer.save(agent, int(average_return), step_cnt)

        scores.append(score)
        episode_cnt += 1
        if episode_cnt % print_interval == 0:
            if hasattr(agent, "buffer"):
                print("Episode: %d, score: %f, buffer capacity: %d" \
                    % (episode_cnt, score, len(agent.buffer)))
            else:
                print("Episode: %d, score: %f" \
                    % (episode_cnt, score))


def PPO_min_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(3e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
    }
    # Generate agent
    agent = PPO(configs)
    # Generate tensorboard writer
    log_writer, ckpt_writer = writer_generator("ppo/min", env_name, "step")
    mujoco_trainer(env, agent, n_step, log_writer=log_writer, ckpt_writer=ckpt_writer)


def PPO_max_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.NormalizeReward(env)
    # Params
    n_step = int(3e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        # "dist_type": "beta",
        "advantage_norm": True,
        "entropy_coef": 0.01,
        # "gradient_clip": True
    }
    # Generate agent
    agent = PPO(configs)
    # Generate tensorboard writer
    log_writer, ckpt_writer = writer_generator("ppo/max", env_name, "step")
    mujoco_trainer(env, agent, n_step, log_writer=log_writer, ckpt_writer=ckpt_writer)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2"
    # env_name = "Ant-v2"
    # PPO_min_mujoco(env_name)
    PPO_max_mujoco(env_name)
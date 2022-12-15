import sys
sys.path.append(".")
sys.path.append("..")
from collections import deque

import numpy as np
import gym

from rllib.algorithms.base.agent import AgentBase
from demo_rl.utils import ActionWrapper


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
            action = agent.get_action(state)
            # action output range[-1,1], expand to allowable range
            action_in =  action_processor.process(action)
            next_state, reward, done, _ = env.step(action_in)
            done_mask = 0.0 if done else 1.0
            if hasattr(agent, "buffer"):
                agent.buffer.push((state, action, reward, done_mask))
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
        
    env.close()


def atari_trainer(
    env: gym.Env, agent: AgentBase, 
    n_step: int = int(5e7), record_interval: int = 1000, print_interval: int = 100,
    log_writer = None, ckpt_writer = None
):

    env_name = env.unwrapped.spec.id

    scores = deque([], maxlen=100)
    step_cnt = 0
    episode_cnt = 0

    while step_cnt < n_step:
        score = 0
        state = env.reset()
        done = False

        while not done:
            state = np.array(state)
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            if hasattr(agent, "buffer"):
                agent.buffer.push((state, action, reward, done_mask))
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
        
    env.close()
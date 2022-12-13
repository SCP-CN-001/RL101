import sys
sys.path.append(".")
sys.path.append("..")
from collections import deque

import numpy as np
import gym

from rllib.algorithms.sac import SAC
from demo_rl.utils import writer_generator, ActionWrapper


def train(
    env: gym.Env, agent: SAC, n_step: int, env_name: str, 
    log_writer = None, ckpt_writer = None
):

    action_processor = ActionWrapper(env.action_space)
    
    scores = deque([], maxlen=100)
    step_cnt = 0
    episode_cnt = 0

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
            agent.buffer.push((state, action, reward, done_mask))
            state = next_state
            score += reward
            agent.train()

            step_cnt += 1
            if step_cnt % 1000 == 0:
                if len(scores) == 0:
                    continue
                average_return = np.mean(scores)
                log_writer.record_scalar("%s_average_return" % env_name, average_return, step_cnt)
                # if step_cnt % 10000 == 0:
                ckpt_writer.save(agent, int(average_return), step_cnt)

        scores.append(score)
        episode_cnt += 1
        if episode_cnt % 100 == 0:
            print("Episode: %d, score: %f, buffer capacity: %d" \
                % (episode_cnt, score, len(agent.buffer)))
        
    env.close()


def SAC_mujoco(env_name):
    # Generate environment
    env = gym.make(env_name)
    # Params
    n_step = int(1e6)
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "buffer_size": int(1e6),
    }
    # Generate agent
    agent = SAC(configs)
    # Record training process

    log_writer, ckpt_writer = writer_generator("sac", env_name, "step")
    train(env, agent, n_step, env_name, log_writer, ckpt_writer)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2" # n_step=int(3e6)
    # env_name = "Ant-v2" # n_step=int(3e6)
    SAC_mujoco(env_name)
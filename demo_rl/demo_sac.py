import sys
sys.path.append(".")
sys.path.append("..")
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
import time

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms.sac import SAC


class ActionProcessor:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_range = [action_space.low, action_space.high]

    def process(self, action):
        action = action * \
            (self.action_range[1] - self.action_range[0]) / 2.0 + \
                (self.action_range[1] + self.action_range[0]) / 2.0
        return action


def train(env: gym.Env, agent: SAC, n_step: int, env_name: str, writer: SummaryWriter):

    action_processor = ActionProcessor(env.action_space)
    
    done = False
    score = 0
    scores = []
    state = env.reset()
    cnt_episode = 0

    for i in range(n_step):
        if done:
            print("Episode: %d, score: %f, buffer capacity: %d" \
                % (cnt_episode, score, len(agent.buffer)))
            cnt_episode +=1
            scores.append(score)
            score = 0
            state = env.reset()
            done = False

        # env.render() # render mujoco environment will take a large part of cpu resource, though it's funny.
        action = agent.get_action(state)
        # action output range[-1,1],expand to allowable range
        action_in =  action_processor.process(action)
        next_state, reward, done, _ = env.step(action_in)
        done_mask = 0.0 if done else 1.0
        agent.buffer.push((state, action, reward, done_mask))
        state = next_state
        score += reward

        agent.train()

        if i % 1000 == 0:
            if len(scores) == 0:
                continue
            else:
                average_return = np.mean(scores)
            print("The average return is %f" % average_return)
            writer.add_scalar("sac_%s_average_return" % env_name, average_return, i)
            scores = []
        
    env.close()


def tensorboard_writer(env_name):
    """Generate a tensorboard writer
    """
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    writer_path = "./logs/sac/%s/%s/" % (env_name, timestamp)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


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
    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, n_step, env_name, writer)


if __name__ == '__main__':
    env_name = "Hopper-v3"
    # env_name = "Walker2d-v3"
    # env_name = "HalfCheetah-v2" # n_step=int(3e6)
    # env_name = "Ant-v2" # n_step=int(3e6)
    # env_name = "Humanoid-v3" # n_step = int(1e7)
    SAC_mujoco(env_name)
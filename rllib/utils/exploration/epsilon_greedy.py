import random
import math

from rllib.algorithms.base.config import ConfigBase


def epsilon_greedy(action, num_step: int, configs: ConfigBase):
    if configs.reduce_epsilon:
        # skip exploration if the maximum exploration step is reached
        if num_step > configs.maximum_exploration_step:
            return action
        # reduce epsilon
        else:
            epsilon_threshold = configs.final_epsilon + \
                (configs.initial_epsilon-configs.final_epsilon) \
                    * math.exp(- num_step * configs.decay_rate)
    else:
        epsilon_threshold = configs.epsilon
    if random.random() < epsilon_threshold:
        return configs.action_space.sample()
    return action
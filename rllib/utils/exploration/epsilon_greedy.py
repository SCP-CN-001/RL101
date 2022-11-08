import random
import math

import torch

from rllib.algorithms.base.config import ConfigBase


def epsilon_greedy(
    action: torch.Tensor, num_step: int, device: torch.device, configs: ConfigBase
) -> torch.Tensor:
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
        return torch.tensor(configs.action_space.sample()).to(device)
    return action
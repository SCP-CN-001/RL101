import random
import math


class EpsilonGreedy(object):
    def __init__(
        self, epsilon: float = 0.1, reduce_epsilon: bool = True, 
        initial_epsilon: float = 1, final_epsilon: float = 0.1, step_decay: int = int(1e6)
    ):
        self.reduce_epsilon = reduce_epsilon
        if self.reduce_epsilon:
            self.initial_epsilon = initial_epsilon
            self.final_epsilon = final_epsilon
            self.step_cnt = 0
            self.step_decay = step_decay
        else:
            self.epsilon = epsilon

    def update_epsilon(self):
        epsilon = self.final_epsilon + \
            (self.initial_epsilon - self.final_epsilon) * math.exp(-1. * self.step_cnt / self.step_decay)
        self.step_cnt += 1
        return epsilon

    def explore(self, action, action_space):
        if self.reduce_epsilon:
            if self.step_cnt < self.step_decay:
                threshold = self.update_epsilon()
            else:
                threshold = self.final_epsilon
        else:
            threshold = self.epsilon
        if random.random() < threshold:
            return action_space.sample()
        return action
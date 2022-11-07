from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, extra_items: list = []):
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.buffer = {}
        for item in self.items:
            self.buffer[item] = deque([], maxlen=buffer_size)
    
    def push(self, observations:tuple):
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.buffer[item].append(observations[i])

    def get_items(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        
        for idx in idx_list:
            for item in self.items:
                batches[item].append(self.buffer[item][idx])
            batches["next_state"].append(self.buffer["state"][idx+1])
        for key in batches.keys():
            batches[key] = np.array(batches[key])
        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__() - 1, size=batch_size)
        return self.get_items(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get_items(idx_list)

    def clear(self):
        for item in self.items:
            self.buffer[item].clear()

    def __len__(self):
        return len(self.buffer["state"])
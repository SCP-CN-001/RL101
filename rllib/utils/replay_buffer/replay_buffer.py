from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, extra_items: list = []):
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.buffer = deque([], maxlen=buffer_size)
    
    def push(self, observations:tuple):
        """Save a transition"""
        self.buffer.append(observations)

    def get_items(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        
        for idx in idx_list:
            for i, item in enumerate(self.items):
                batches[item].append(self.buffer[idx][i])
            batches["next_state"].append(self.buffer[idx+1][0])

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

    def all(self):
        idx_range = self.__len__() - 1
        idx_list = np.arange(idx_range)
        return self.get_items(idx_list)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
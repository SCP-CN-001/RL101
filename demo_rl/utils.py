import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter


def writer_generator(
    model_name:str, env_name: str,
    ckpt_prefix: str, num_ckpt: int = 10
):
    current_path = os.getcwd()
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    log_path = "%s/logs/%s/%s/%s/" % (current_path, model_name, env_name, timestamp)
    ckpt_path = "%s/models/%s/%s/%s/" % (current_path, model_name, env_name, timestamp)
    log_writer = LogWriter(log_path)
    ckpt_writer = CheckpointWriter(ckpt_path, ckpt_prefix, num_ckpt)
    return log_writer, ckpt_writer


class LogWriter:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.writer = SummaryWriter(self.path)

    def record_scalar(self, *args):
        self.writer.add_scalar(*args)


class CheckpointWriter:
    def __init__(self, path: str, prefix: str, num_ckpt: int = 10):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.prefix = prefix
        self.num_ckpt = num_ckpt
        self.ckpt_idx = []

    def save(self, agent, average_return: int, suffix: int):

        self.ckpt_idx.append((average_return, suffix))
        self.ckpt_idx.sort(key=lambda idx: idx[0], reverse=True)

        if len(self.ckpt_idx) <= self.num_ckpt:
            path = "%s/%s_%d.pt" % (self.path, self.prefix, suffix)
            agent.save(path)
        else:
            remove_idx = self.ckpt_idx.pop()
            if remove_idx[1] != suffix:
                save_path = "%s/%s_%d.pt" % (self.path, self.prefix, suffix)
                agent.save(save_path)
                remove_path = "%s/%s_%d.pt" % (self.path, self.prefix, remove_idx[1])
                os.remove(remove_path)


class ActionWrapper:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_range = [action_space.low, action_space.high]

    def process(self, action):
        action = action * \
            (self.action_range[1] - self.action_range[0]) / 2.0 + \
                (self.action_range[1] + self.action_range[0]) / 2.0
        return action
import os


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

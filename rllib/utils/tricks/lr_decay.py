def lr_decay(self, lr: float, n_step: int, decay_type: str = "exp") -> float:
    if decay_type == "linear":
        lr = lr * (1 - n_step / self.configs.max_train_steps)
    elif decay_type == "exp":
        lr = lr * np.exp(-n_step / self.configs.max_train_steps)
    return lr
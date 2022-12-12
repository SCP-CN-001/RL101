import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim: int, mu: float, theta: float, sigma: float, step: float):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.step = step
        self.reset()

    def __call__(self) -> np.array:
        x = self.x
        dx = self.theta * (self.mu - x) + \
            self.sigma * np.random.normal(size=self.x.shape)
        self.x = x + dx
        return self.x * self.step

    def reset(self):
        self.x = np.ones(self.action_dim) * self.mu
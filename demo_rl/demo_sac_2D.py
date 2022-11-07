import sys
sys.path.append(".")
sys.path.append("..")
from typing import Tuple

import numpy as np
import gym
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils import tensorboard

from rllib.algorithms.sac import SACActor, SACCritic, SAC


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActorConv2D(nn.Module):
    def __init__(
        self, log_std_min: float, log_std_max: float, epsilon: float
    ):
        super(ActorConv2D, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.mean_layer = nn.Linear(200, 3)
        self.log_std_layer = nn.Linear(200, 3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.net(state)
        x = x.view(-1, 200)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state: torch.Tensor)  -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action[0]

    def evaluate(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        """Implement the re-parameterization trick f()
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        z = z.to(device)
        action = torch.tanh(mean + std * z)
        log_prob = normal.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + self.epsilon)

        return action, log_prob


class CriticConv2D(nn.Module):
    def __init__(self):
        super(CriticConv2D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc= nn.Linear(5120, 3)


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action[:, :, None, None]
        action = action.repeat(1,1,96,96)
        x = torch.cat([state, action], 3)
        x = self.net(x)
        print(x.size())
        x.view(-1, 56320)
        print(x.size())
        x = self.fc(x)
        print(x.size())
        return x


def main(env, agent, episode):
    action_range = [env.action_space.low, env.action_space.high]
    rewards = []

    for episode in range(episode):
        score = 0
        state = env.reset()
        for i in range(300):
            state = state.copy()
            state = np.transpose(state, (2,0,1))

            action = agent.get_action(state)
            action_in =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
            next_state, reward, done, _ = env.step(action_in)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, done_mask))
            state = next_state
            score += reward
            env.render()
            if done:
                break
            if len(agent.buffer) > 500:
                agent.train()

        print("episode:{}, score:{}, buffer_capacity:{}".format(episode, score, len(agent.buffer)))
        rewards.append(score)
    return

if __name__ == "__main__":
    env = gym.make("CarRacing-v0")
    configs = {
        "state_space": env.observation_space,
        "state_dim": env.observation_space.shape,
        "action_space": env.action_space,
        "action_dim": env.observation_space.shape[0],
        "actor_net": ActorConv2D,
        "actor_kwargs": {
            "log_std_min": -20,
            "log_std_max": 2,
            "epsilon": 1e-6
        },
        "critic_net": CriticConv2D,
        "critic_kwargs": {},
    }

    episode = 100
    agent = SAC(configs)
    main(env, agent, episode)
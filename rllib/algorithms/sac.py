from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from rllib.algorithms.base.agent import AgentBase
from rllib.algorithms.base.config import ConfigBase
from rllib.replay_buffer.replay_buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SACActor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size: int, 
        log_std_min: float, log_std_max: float, epsilon: float
    ):
        super(SACActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state: torch.Tensor)  -> np.ndarray:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    def evaluate(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        """Implement the re-parameterization trick f()
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        z = z.to(device)
        action = torch.tanh(mean + std * z)
        log_prob = dist.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + self.epsilon)

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(SACCritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.net(x)

        return x


class SACConfig(ConfigBase):
    """Configuration of the SAC model
    """
    def __init__(self, configs: dict):
        super().__init__()

        for key in ["state_space", "action_space"]:
            if key in configs:
                setattr(self, key, configs[key])
            else:
                raise AttributeError("[%s] is not defined for SACConfig!" % key)
        if "state_dim" not in configs.keys():
            self.state_dim = self.state_space.shape[0]
        else:
            self.state_dim = configs["state_dim"]
        if "action_dim" not in configs.keys():
            self.action_dim = self.action_space.shape[0]
        else:
            self.action_dim = configs["action_dim"]

        # model
        ## hyper-parameters
        self.gamma: float = 0.99
        self.batch_size: int = 128
        self.tau: float = 5e-3          # soft-update factor
        self.buffer_size: int = int(1e6)

        ## actor net
        self.lr_actor = 3e-4
        self.actor_net = SACActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256,
            "log_std_min": -20,
            "log_std_max": 2,
            "epsilon": 1e-6
        }

        ## critic net
        self.lr_critic = 3e-4
        self.critic_net = SACCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256
        }

        ## alpha
        self.lr_alpha = 3e-4
        self.learn_temperature = True
        self.initial_temperature = 0.2
        self.target_entropy = -self.action_dim

        # tricks

        self.merge_configs(configs)


class SAC(AgentBase):
    """Soft Actor-Critic (SAC)
    An implementation of SAC based on the 2nd version of SAC paper 'Soft Actor-Critic Algorithms and Applications'
    """
    def __init__(self, configs: dict):
        super().__init__(SACConfig, configs)

        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(device)

        ## critic nets
        self.critic_net1 = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target_net1 = deepcopy(self.critic_net1)

        self.critic_net2 = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target_net2 = deepcopy(self.critic_net2)

        ## alpha
        self.log_alpha = torch.tensor(np.log(self.configs.initial_temperature)).to(device)
        self.log_alpha.requires_grad = True

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer1 = torch.optim.Adam(
            self.critic_net1.parameters(), self.configs.lr_critic)
        self.critic_optimizer2 = torch.optim.Adam(
            self.critic_net2.parameters(), self.configs.lr_critic)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], self.configs.lr_alpha)
        
        # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        action = self.actor_net.action(state)

        return action

    def soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.configs.tau + target.data * (1. - self.configs.tau))

    def train(self):
        if len(self.buffer) < self.configs.batch_size:
            return

        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(device)
        action = torch.FloatTensor(batches["action"]).to(device)
        reward = torch.FloatTensor(batches["reward"]).unsqueeze(-1).to(device)
        next_state = torch.FloatTensor(batches["next_state"]).to(device)
        done = torch.FloatTensor(batches["done"]).unsqueeze(-1).to(device)

        # soft Q loss
        next_action, next_log_prob = self.actor_net.evaluate(next_state)
        next_log_prob = next_log_prob.sum(-1, keepdim=True)
        q1_target = self.critic_target_net1(next_state, next_action)
        q2_target = self.critic_target_net2(next_state, next_action)
        q_target = reward + done * self.configs.gamma * (torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_prob)

        current_q1 = self.critic_net1(state, action)
        current_q2 = self.critic_net2(state, action)
        q1_loss = F.mse_loss(current_q1, q_target.detach())
        q2_loss = F.mse_loss(current_q2, q_target.detach())

        # update the critic networks
        self.critic_optimizer1.zero_grad()
        q1_loss.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.zero_grad()
        q2_loss.backward()
        self.critic_optimizer2.step()

        # policy loss
        action_, log_prob = self.actor_net.evaluate(state)
        log_prob = log_prob.sum(-1, keepdim=True)
        q1_value = self.critic_net1(state, action_)
        q2_value = self.critic_net2(state, action_)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_value, q2_value)).mean()

        # update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # optimize alpha
        if self.configs.learn_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.configs.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        # soft update target networks
        self.soft_update(self.critic_target_net1, self.critic_net1)
        self.soft_update(self.critic_target_net2, self.critic_net2)

    def save(self, path: str):
        torch.save({
            "actor_net": self.actor_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_net1": self.critic_net1.state_dict(),
            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_net2": self.critic_net2.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
            "log_alpha": self.log_alpha,
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])
        self.log_alpha = checkpoint["log_alpha"]
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        self.critic_target_net1 = deepcopy(self.critic_net1)
        self.critic_target_net2 = deepcopy(self.critic_net2)
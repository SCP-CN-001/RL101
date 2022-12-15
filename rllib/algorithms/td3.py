from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.replay_buffer.replay_buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TD3Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(TD3Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)

        return x

    def action(self, state: torch.Tensor)  -> np.ndarray:
        x = self.forward(state)
        action = torch.tanh(x)

        return action


class TD3Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(TD3Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.net(x)

        return x


class TD3Config(ConfigBase):
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
        self.batch_size = 100
        self.tau: float = 5e-3         # soft-update factors
        self.buffer_size: int = int(1e6)
        self.explore_noise_sigma = 0.1
        self.explore_noise_clip = 0.2
        self.tps_noise_sigma = 0.2
        self.tps_noise_clip = 0.5
        self.target_update_freq = 2

        ## actor net
        self.lr_actor = 1e-3
        self.actor_net = TD3Actor
        self.actor_kwargs= {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256
        }

        ## critic net
        self.lr_critic = 1e-3
        self.critic_net = TD3Critic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 256,
        }


class TD3(AgentBase):
    def __init__(self, configs: dict):
        super().__init__(TD3Config, configs)

        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(device)
        self.actor_target_net = deepcopy(self.actor_net)
        self.update_cnt = 0

        ## critic net
        self.critic_net1 = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target_net1 = deepcopy(self.critic_net1)

        self.critic_net2 = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target_net2 = deepcopy(self.critic_net2)

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer1 = torch.optim.Adam(
            self.critic_net1.parameters(), self.configs.lr_critic)
        self.critic_optimizer2 = torch.optim.Adam(
            self.critic_net2.parameters(), self.configs.lr_critic)

        # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

        # exploration
        self.explore_noise_sigma = self.configs.explore_noise_sigma
        self.explore_noise_clip = self.configs.explore_noise_clip

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        action = self.actor_net.action(state)

        # explore
        noise = torch.randn_like(action) * self.explore_noise_sigma
        noise = noise.clamp(-self.explore_noise_clip, self.explore_noise_clip)
        action += noise
        action = action.clamp(-1, 1)
        action = action.detach().cpu().numpy()

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

        # target policy smoothing
        noise = torch.randn_like(action) * self.configs.tps_noise_sigma
        noise = noise.clamp(-self.configs.tps_noise_clip, self.configs.tps_noise_clip)
        next_action = self.actor_target_net.action(next_state) + noise
        next_action = next_action.clamp(-1, 1)

        # critic loss
        q1_target = self.critic_target_net1(next_state, next_action)
        q2_target = self.critic_target_net2(next_state, next_action)
        q_target = reward + done * self.configs.gamma * torch.min(q1_target, q2_target)

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

        # delayed policy updates
        self.update_cnt += 1
        if self.update_cnt % self.configs.target_update_freq == 0:
            action_ = self.actor_net.action(state)
            q1_value = self.critic_net1(state, action_)
            actor_loss = - q1_value.mean()
            # update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # soft update target networks
            self.soft_update(self.actor_target_net, self.actor_net)
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
        }, path)

    def load(self, path: str, map_location = None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])

        self.actor_target_net = deepcopy(self.actor_net)
        self.critic_target_net1 = deepcopy(self.critic_net1)
        self.critic_target_net2 = deepcopy(self.critic_net2)
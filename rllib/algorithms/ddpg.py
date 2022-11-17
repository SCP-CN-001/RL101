from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.utils.replay_buffer.replay_buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DDPGActor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size1: int, hidden_size2: int,
        init_weight: float
    ):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_dim),
            nn.Tanh()
        )
        self.net[4].weight.data.uniform_(-init_weight, init_weight)
        self.net[4].bias.data.uniform_(-init_weight, init_weight)

    def forward(self, state: torch.Tensor):
        x = self.net(state)
        return x


class DDPGCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size1: int, hidden_size2: int,
        init_weight: float
    ):
        super(DDPGCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.net(x)

        return x


class DDPGConfig(ConfigBase):
    def __init__(self, configs: dict):
        super().__init__()
        # The default parameters are referred to the original paper's setting in the low dimension scenarios.

        # model
        ## hyper-parameters
        self.gamma: float = 0.99
        self.batch_size = 64
        self.tau: float = 1e-3         # soft-update factors
        self.buffer_size: int = int(1e6)

        ## actor net
        self.lr_actor = 1e-4
        self.actor_net = DDPGActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300
        }

        ## critic net
        self.lr_critic = 1e-3
        self.critic_net = DDPGCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300
        }

        self.merge_configs(configs)


class DDPG(AgentBase):
    """_summary_
    """
    def __init__(self, configs: dict):
        super().__init__(DDPGConfig, configs)
        
        # networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(device)
        self.actor_target_net = deepcopy(self.actor_net)

        ## critic net
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target_net = deepcopy(self.critic_net)

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), self.configs.lr_critic_net)

        # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
    
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
        
        # Q target
        next_action = self.actor_target_net(state)
        q_next = self.critic_target_net(next_state, next_action)
        q_target = reward + done * self.configs.gamma * q_next

        # Q value
        q_value = self.critic_net(state, action)

        # update the actor network
        actor_loss = -self.critic_net(state, action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the critic network
        critic_loss = F.mse_loss(q_value, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft update target networks
        self.soft_update(self.actor_target_net, self.actor_net)
        self.soft_update(self.critic_target_net, self.critic_net)
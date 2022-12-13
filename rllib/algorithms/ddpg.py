from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.replay_buffer.replay_buffer import ReplayBuffer
from rllib.exploration.ornstein_uhlenbeck_noise import OrnsteinUhlenbeckNoise


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def linear_fanin(linear_layer: nn.Linear):
    return linear_layer.weight.data.size()[0]


class DDPGActor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size1: int, hidden_size2: int,
        init_weight: float
    ):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # initial weight and bias
        fc1_fanin = linear_fanin(self.fc1)
        self.fc1.weight.data.uniform_(-1/np.sqrt(fc1_fanin), 1/np.sqrt(fc1_fanin))
        self.fc1.bias.data.uniform_(-1/np.sqrt(fc1_fanin), 1/np.sqrt(fc1_fanin))
        fc2_fanin = linear_fanin(self.fc2)
        self.fc2.weight.data.uniform_(-1/np.sqrt(fc2_fanin), 1/np.sqrt(fc2_fanin))
        self.fc2.bias.data.uniform_(-1/np.sqrt(fc2_fanin), 1/np.sqrt(fc2_fanin))
        self.fc3.weight.data.uniform_(-init_weight, init_weight)
        self.fc3.bias.data.uniform_(-init_weight, init_weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

    def action(self, state: torch.Tensor)  -> np.ndarray:
        x = self.forward(state)
        action = x.detach().cpu().numpy()
        return action


class DDPGCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_size1: int, hidden_size2: int,
        init_weight: float
    ):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()

        # initial weight and bias
        fc1_fanin = linear_fanin(self.fc1)
        self.fc1.weight.data.uniform_(-1/np.sqrt(fc1_fanin), 1/np.sqrt(fc1_fanin))
        self.fc1.bias.data.uniform_(-1/np.sqrt(fc1_fanin), 1/np.sqrt(fc1_fanin))
        fc2_fanin = linear_fanin(self.fc2)
        self.fc2.weight.data.uniform_(-1/np.sqrt(fc2_fanin), 1/np.sqrt(fc2_fanin))
        self.fc2.bias.data.uniform_(-1/np.sqrt(fc2_fanin), 1/np.sqrt(fc2_fanin))
        self.fc3.weight.data.uniform_(-init_weight, init_weight)
        self.fc3.bias.data.uniform_(-init_weight, init_weight)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DDPGConfig(ConfigBase):
    """Configuration of the DDPG model
    """
    def __init__(self, configs: dict):
        super().__init__()

        for key in ["state_space", "action_space"]:
            if key in configs:
                setattr(self, key, configs[key])
            else:
                raise AttributeError("[%s] is not defined for DDPGConfig!" % key)
        if "state_dim" not in configs.keys():
            self.state_dim = self.state_space.shape[0]
        else:
            self.state_dim = configs["state_dim"]
        if "action_dim" not in configs.keys():
            self.action_dim = self.action_space.shape[0]
        else:
            self.action_dim = configs["action_dim"]

        # The default parameters are referred to the original paper's setting in the low dimension scenarios.

        # model
        ## hyper-parameters
        self.gamma: float = 0.99
        self.batch_size = 64
        self.tau: float = 1e-3         # soft-update factors
        self.buffer_size: int = int(1e6)
        self.ou_theta = 0.15          # for exploration based on Ornstein–Uhlenbeck process
        self.ou_sigma = 0.2
        self.ou_step = 0.002

        ## actor net
        self.lr_actor = 1e-4
        self.actor_net = DDPGActor
        self.actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300,
            "init_weight": 3e-3
        }

        ## critic net
        self.lr_critic = 1e-3
        self.critic_net = DDPGCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size1": 400,
            "hidden_size2": 300,
            "init_weight": 3e-4
        }

        self.merge_configs(configs)


class DDPG(AgentBase):
    """Deep Deterministic Policy Gradient (DDPG)
    An implementation of DDPG based on the original paper 'Continuous control with deep reinforcement learning'
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
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), self.configs.lr_critic)

        # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

        # exploration
        self.noise_generator = OrnsteinUhlenbeckNoise(
            self.configs.action_dim, 
            0, self.configs.ou_theta, self.configs.ou_sigma, self.configs.ou_step,
        )

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        action = self.actor_net.action(state)
        action += self.noise_generator()

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

        # update the critic network
        next_action = self.actor_target_net(state)
        q_next = self.critic_target_net(next_state, next_action)
        q_target = reward + done * self.configs.gamma * q_next
        q_value = self.critic_net(state, action)
        critic_loss = F.mse_loss(q_value, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the actor network
        actor_loss = -self.critic_net(state, self.actor_net(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update target networks
        self.soft_update(self.actor_target_net, self.actor_net)
        self.soft_update(self.critic_target_net, self.critic_net)

    def save(self, path: str):
        torch.save({
            "actor_net": self.actor_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.actor_target_net = deepcopy(self.actor_net)
        self.critic_target_net = deepcopy(self.critic_net)
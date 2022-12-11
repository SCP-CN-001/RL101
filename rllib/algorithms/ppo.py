from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.utils.replay_buffer.replay_buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PPOActor(nn.Module):
    def __init__(
        self, discrete: bool, state_dim: int, action_dim: int, hidden_size: int
    ):
        super(PPOActor, self).__init__()
        self.discrete = discrete
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.softmax = nn.Softmax(-1)
        
    def forward(self, state: torch.Tensor):
        x = self.net(state)
        mean = self.mean_layer(x)

        return mean

    def action(self, state: torch.Tensor) -> np.ndarray:
        if self.discrete:
            probs = self.forward(state)
            probs = self.softmax(probs)
            dist = Categorical(probs)
        else:
            dist = self.get_norm_dist(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def get_norm_dist(self, state: torch.Tensor):
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        return dist

class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int):
        super(PPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)

        return x


class PPOConfig(ConfigBase):
    """Configuration of the PPO model
    """
    def __init__(self, configs):
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
        self.discrete = configs["discrete"] if "discrete" in configs.keys() else False
        self.horizon = 2048
        self.epoch = 10
        self.batch_size = 64
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2

        ## actor net
        self.lr_actor = self.lr
        self.actor_net = PPOActor
        self.actor_kwargs = {
            "discrete": self.discrete,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": 64,
        }

        ## critic net
        self.lr_critic = self.lr
        self.critic_net = PPOCritic
        self.critic_kwargs = {
            "state_dim": self.state_dim,
            "hidden_size": 64,
        }

        # tricks
        ## By default none of the tricks are used, and the performance of the 
        ## minimalism-style PPO is not very beautiful. 

        # advantage normalization
        self.advantage_norm = False
        # policy entropy
        self.entropy_coef = None
        # gradient clipping
        self.gradient_clip = False
        self.gradient_clip_range = 0.5
        
        self.merge_configs(configs)

class PPO(AgentBase):
    """Proximal Policy Optimization (PPO)
    An implementation of PPO based on the original paper 'Proximal Policy Optimization Algorithms'. 

    The performance of the minimal PPO is not good, so the tricks from 'Implementation 
    Matters in Deep Policy Gradients: A Case Study on PPO and TRPO' are also implemented 
    and controlled by the PPOConfig.
    """
    def __init__(self, configs: dict) -> None:
        super().__init__(PPOConfig, configs)

        # the networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(device)

        ## critic nets
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(device)

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), self.configs.lr_critic)

        ## buffer
        self.buffer = ReplayBuffer(
            self.configs.horizon, extra_items=["next_state", "log_prob"]
        )

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        
        action, log_prob = self.actor_net.action(state)
        return action, log_prob

    def train(self):
        if len(self.buffer) < self.configs.horizon:
            return

        batches = self.buffer.all()
        state = torch.FloatTensor(batches["state"]).to(device)
        action = torch.FloatTensor(batches["action"]).to(device)
        reward = torch.FloatTensor(batches["reward"]).unsqueeze(-1).to(device)
        done = torch.FloatTensor(batches["done"]).unsqueeze(-1).to(device)
        next_state = torch.FloatTensor(batches["next_state"]).to(device)
        old_log_prob = torch.FloatTensor(batches["log_prob"]).to(device)

        if not self.configs.advantage_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        # generalized advantage estimation
        gae = 0
        advantage = []
        with torch.no_grad():
            value =  self.critic_net(state)
            value_next = self.critic_net(next_state)
            deltas = reward + self.configs.gamma * done * value_next - value
            for delta, is_done in \
                zip(reversed(deltas.detach().cpu().numpy()), reversed(done.detach().cpu().numpy())):
                gae = delta + self.configs.gamma * is_done * self.configs.gae_lambda * gae
                advantage.append(gae)
            advantage.reverse()
            advantage = torch.FloatTensor(np.array(advantage)).to(device)

        v_target = advantage + value

        if self.configs.advantage_norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        for _ in range(self.configs.epoch):
            for indices in BatchSampler(
                SubsetRandomSampler(range(self.configs.horizon-1)), 
                self.configs.batch_size, True
            ):
                if self.configs.discrete:
                    probs = self.actor_net(state)
                    probs = self.softmax(probs)
                    dist = Categorical(probs)
                else:
                    dist = self.actor_net.get_norm_dist(state[indices])
                log_prob = dist.log_prob(action[indices])
                ratio = torch.exp(log_prob.sum(1, keepdim=True) - old_log_prob[indices].sum(1, keepdim=True))

                loss_cpi = ratio * advantage[indices]
                loss_clip = torch.clamp(ratio, 1-self.configs.clip_epsilon, 1+self.configs.clip_epsilon) * advantage[indices]

                if self.configs.entropy_coef is None:
                    actor_loss = - torch.min(loss_cpi, loss_clip).mean()
                else:
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    actor_loss = (- torch.min(loss_cpi, loss_clip) - self.configs.entropy_coef * dist_entropy).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.configs.gradient_clip:
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.configs.gradient_clip_range)
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(self.critic_net(state[indices]), v_target[indices])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.configs.gradient_clip:
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.configs.gradient_clip_range)
                self.critic_optimizer.step()

        self.buffer.clear()
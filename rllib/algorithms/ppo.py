from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(PPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        return mean, log_std

    def action(self, state: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        z =dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action


class PPOCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):

        return


class PPOConfig(ConfigBase):
    def __init__(self, configs):
        super().__init__()
        self.discrete = False

        # model
        ## hyper-parameters
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2

        ## actor net
        self.lr_actor = self.lr
        self.actor_net = PPOActor
        self.actor_kwargs = {
            
        }

        ## critic net
        self.lr_critic = self.lr
        self.critic_net = PPOCritic
        self.critic_kwargs = {

        }

        self.adam_epsilon = 1e-8
        
        self.lambda_ = 0.95
        self.var_max = 1

        # tricks
        self.adv_norm = True
        self.state_norm = True
        self.reward_norm = False
        self.reward_scaling = False
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01

        self.merge_configs(configs)


class PPO(AgentBase):
    """Proximal Policy Optimization (PPO)
    An implementation of PPO based on paper 'Proximal Policy Optimization Algorithms'
    """
    def __init__(self, configs: dict) -> None:

        super().__init__(PPOConfig, configs)

        # the networks
        ## actor net
        self.actor_net = self.configs.actor_net(**self.configs.actor_kwargs).to(device)

        ## critic nets
        self.critic_net = self.configs.critic_net(**self.configs.critic_kwargs).to(device)
        self.critic_target = deepcopy(self.critic_net)

        ## optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), self.configs.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), self.configs.lr_critic)


    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        action = self.actor_net.action(state)

        return action

    def soft_update(self, target_net, current_net):
        for target, current in zip(target_net.parameters(), current_net.parameters()):
            target.data.copy_(current.data * self.configs.tau + target.data * (1. - self.configs.tau))

    def train(self):

        batches = self.memory.shuffle()
        state_batch = torch.FloatTensor(batches["state"]).to(self.device)
        if self.discrete:
            state_batch = torch.IntTensor(state_batch)
        action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        rewards = self._reward_norm(batches["reward"]) \
            if self.configs.reward_norm else batches["reward"] 
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(batches["done"]).to(self.device)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        next_state_batch = torch.FloatTensor(batches["next_state"]).to(self.device)

        # GAE
        gae = 0
        adv = []

        with torch.no_grad():
            value = self.critic_net(state_batch)
            next_value = self.critic_net(next_state_batch)
            deltas = reward_batch + self.configs.gamma * (1 - done_batch) * next_value - value
            for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                gae = delta + self.configs.gamma * self.configs.lambda_ * gae * (1.0 - done)
                adv.append(gae)
            adv.reverse()
            adv = torch.FloatTensor(adv).view(-1, 1).to(self.device)
            v_target = adv + value
            if self.configs.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        for i in range(self.configs.batch_size):
            state = state_batch[i].unsqueeze(0)
            if self.discrete:
                dist = Categorical(self.actor_net(state))
                dist_entropy = dist.entropy().view(-1, 1)
                log_prob= dist.log_prob(action_batch[i].squeeze()).view(-1, 1)
            elif self.configs.dist_type == "beta":
                policy_dist = self.actor_net(state)
                alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                log_prob = dist.log_prob(action_batch[i])
            elif self.configs.dist_type == "gaussian":
                policy_dist = self.actor_net(state)
                mean = torch.clamp(policy_dist, -1, 1)
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                log_prob = torch.sum(dist.log_prob(action_batch[i]))
            prob_ratio = (log_prob - old_log_prob_batch[i]).exp()

            loss1 = prob_ratio * adv[i]
            loss2 = torch.clamp(prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon) * adv[i]

            actor_loss = - torch.min(loss1, loss2)
            if self.configs.policy_entropy:
                actor_loss += - self.configs.entropy_coef * dist_entropy
            critic_loss = F.mse_loss(v_target[i], self.critic_net(state))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.mean().backward()
            critic_loss.mean().backward()
            if self.configs.gradient_clip: # gradient clip
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step() 

        self.soft_update(self.critic_target, self.critic_net)
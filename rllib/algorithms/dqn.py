from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rllib.algorithms.base.config import ConfigBase
from rllib.algorithms.base.agent import AgentBase
from rllib.utils.replay_buffer.replay_buffer import ReplayBuffer
from rllib.utils.exploration.epsilon_greedy import EpsilonGreedy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def action(self, state: torch.Tensor) -> int:
        x = self.forward(state)
        action = x.max(1)[1].view(1,1)
        action = action.detach().cpu().numpy()[0][0]

        return action


class DQNConfig(ConfigBase):
    """Configuration of the DQN model
    """
    def __init__(self, configs: dict):
        super().__init__()

        for key in ["state_space", "action_space"]:
            if key in configs:
                setattr(self, key, configs[key])
            else:
                raise AttributeError("[%s] is not defined for DQNConfig!" % key)
        if "action_dim" not in configs.keys():
            self.num_actions = self.action_space.n
        else:
            self.num_actions = configs["action_dim"]

        # model
        ## hyper-parameters
        self.explore = True
        self.explore_func = EpsilonGreedy()
        self.replay_start_size = 5e4
        self.buffer_size: int = int(1e6)

        ## networks
        self.lr = 2.5e-4
        self.q_net = QNetwork
        self.q_net_kwargs = {
            "num_actions": self.num_actions
        }
        self.target_update_freq = 1e4

        # tricks
        self.gradient_clip = False
        self.gradient_clip_range = 1

        self.merge_configs(configs)


class DQN(AgentBase):
    """Deep Q-Networks (DQN)
    An implementation of DQN based on the Nature released version of DQN paper 'Human-level control through deep reinforcement learning'
    """
    def __init__(self, configs: dict):
        super().__init__(DQNConfig, configs)

        # networks
        self.policy_net = self.configs.q_net(**self.configs.q_net_kwargs).to(device)
        self.target_net = deepcopy(self.policy_net)
        self.learn_step_cnt = 0

        # optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.configs.lr)

        # the replay buffer
        self.buffer = ReplayBuffer(self.configs.buffer_size)

        # exploration method
        if self.configs.explore:
            self.explore_func = self.configs.explore_func

    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)

        state = state.unsqueeze(0)
        action = self.policy_net.action(state)
        if self.configs.explore:
            action = self.explore_func.explore(action, self.configs.action_space)
        
        return action
    
    def train(self):
        if len(self.buffer) < self.configs.replay_start_size:
            return
        
        batches = self.buffer.sample(self.configs.batch_size)
        state = torch.FloatTensor(batches["state"]).to(device)
        action = torch.LongTensor(batches["action"]).to(device)
        reward = torch.FloatTensor(batches["reward"]).to(device)
        next_state = torch.FloatTensor(batches["next_state"]).to(device)
        done = torch.FloatTensor(batches["done"]).to(device)

        # loss function
        q_value = self.policy_net(state).gather(1, action)
        q_next = self.target_net(next_state).max(1)[0].detach()
        q_target = (reward + q_next * self.configs.gamma * done).unsqueeze(-1)
        loss = F.mse_loss(q_value, q_target)

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        if self.configs.gradient_clip:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.configs.gradient_clip_range)
        self.optimizer.step()

        # update target net
        self.learn_step_cnt += 1
        if self.learn_step_cnt % self.configs.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
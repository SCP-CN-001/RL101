import os

import torch
from torch import optim
import torch.nn.functional as F


class DQN(object):
    """The original DQN model"""
    # The implementation details have referred to 
    # Roderick, Melrose, James MacGlashan, and Stefanie Tellex. "Implementing the deep q-network." arXiv preprint arXiv:1711.07478 (2017).
    def __init__(self, config, hyperparameter):
        self.config = config
        self.hyperparameter = hyperparameter

        self.q_network = None
        
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_decay = config.decay
        
        self.n_step = 0

    def act(self, observation):
        
        return self.action
    
    def compute_q_target(self, next_states, reward):
        pass
    
    def q_value_current_state(self, reward):
        pass
    
    def q_value_next_state(self):
        pass
    
    def loss(self):
        with torch.no_grad():
            Q_target = self.q_value_next_state()
        Q_expected = self.q_value_current_state()
        loss = F.mse_loss(Q_expected, Q_target)
        return loss
    
    def to_learn(self):
        """Return a boolean indicating whether it is time to update policy"""
        
        return
    
    def learn(self, reward, episode):
        loss = self.loss(reward)
        
        self.save_param("current")
        if self.save:
            if self.n_save % episode == 0:
                self.save_param(str(episode))
            
        
    def save_param(self, suffix):
        torch.save(self.network.state_dict(), "%s/episode_%s.pkl" % (self.dir_save, suffix))
        
    def load_param(self, suffix):
        self.network.load_state_dict(torch.load("%s/episode_%s.pkl" % (self.dir_save, suffix)))
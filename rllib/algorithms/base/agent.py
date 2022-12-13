from abc import ABC, abstractmethod

from rllib.algorithms.base.config import ConfigBase


class AgentBase(ABC):
    def __init__(self, config_type: ConfigBase, configs:dict):
        """Initialize the model structure here
        """
        self.configs = config_type(configs)

    @abstractmethod
    def get_action(self, state):
        """Return an action based on the input state
        """

    @abstractmethod
    def train(self):
        """Update the network's parameters.
        """

    @abstractmethod
    def save(self, path: str):
        """Save the networks to the given path.
        """

    @abstractmethod
    def load(self, path: str):
        """Load the networks from the given path.
        """
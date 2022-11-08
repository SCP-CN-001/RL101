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

    def save_checkpoint(self, checkpoint_dir: str):
        """Store the model structure and corresponding parameters to a file.
        """
        return

    def load_checkpoint(self, path: str = None, params_only: bool = None) -> None:
        """Load the model structure and corresponding parameters from a file.
        """
        return
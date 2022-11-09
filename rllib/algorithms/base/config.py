class ConfigBase:
    def __init__(self):
        # runtime
        self.n_epoch = None
        self.n_initial_exploration_steps = None

        # environment
        self.state_space = None
        self.action_space = None
        self.state_dim = None
        self.action_dim = None

        # model
        self.gamma: float = 0.99    # reward discount factor
        self.batch_size:int = 128   # batch size
        self.lr: float = 1e-3           # learning rate

        # explore
        self.explore: bool = True
        self.explore_config: dict = {
            "type": "StochasticSampling",
        }

        # evaluation
        self.evaluation_interval = None
        self.evaluation_duration = 10
        self.evaluation_duration_unit = "episodes"

        # debug
        self.log_level = "verbose"
        self.seed = None

    def merge_configs(self, configs: dict):
        """Merge the custom configs for a specific algorithm

        Args:
            configs (dict): the custom configs
        """
        for key, value in configs.items():
            setattr(self, key, value)
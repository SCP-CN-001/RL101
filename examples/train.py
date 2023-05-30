import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../rllib")

import argparse

import numpy as np
import gymnasium as gym
import torch

from examples.trainer import trainer
from rllib.algorithms import ddpg, td3, sac, ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--max-step", type=int, default=int(3e6))
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--clip-action", action="store_true", default=False)
    # parser.add_argument("--normalize-observation", action="store_true", default=False)
    args = parser.parse_args()

    env = gym.make(args.env)
    device = torch.device(args.device)

    if args.agent == "ddpg":
        agent_config = ddpg.DDPGConfig(
            {"state_space": env.observation_space, "action_space": env.action_space}
        )
        agent = ddpg.DDPG(agent_config, device)
    elif args.agent == "td3":
        agent_config = td3.TD3Config(
            {"state_space": env.observation_space, "action_space": env.action_space}
        )
        agent = td3.TD3(agent_config, device)
    elif args.agent == "sac":
        agent_config = sac.SACConfig(
            {"state_space": env.observation_space, "action_space": env.action_space}
        )
        agent = sac.SAC(agent_config, device)
    elif args.agent == "ppo":
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        agent_config = ppo.PPOConfig(
            {"state_space": env.observation_space, "action_space": env.action_space}
        )
        agent = ppo.PPO(agent_config, device)


    trainer(env, agent, max_step=args.max_step, log_tag=f"{args.tag}/{args.env}")

import sys
sys.path.append(".")
sys.path.append("..")
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
import time

import numpy as np
from PIL import Image
import gym
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms.dqn import DQN


np.random.seed(20)


def tensorboard_writer(env_name):
    """Generate a tensorboard writer
    """
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    writer_path = "./logs/dqn/%s/%s/" % (env_name, timestamp)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


def process_atari(state):
    im = Image.fromarray(state, mode="RGB")
    im = im.resize((84, 84))
    gray = im.convert('L')
    im_array = np.array(im)
    gray_array = np.expand_dims(np.array(gray), axis=2)
    processed_state = np.concatenate((im_array, gray_array), axis=2)
    processed_state = np.transpose(processed_state, (2, 0, 1))
    processed_state = processed_state / 255.
    return processed_state


def train(
    env: gym.Env, agent:DQN, num_episode: int, update_freq: int,
    env_name: str, writer: SummaryWriter, preprocess_state = None
):
    counter = 0

    for episode in range(num_episode):
        score = 0
        state = env.reset()
        is_done = False
        while not is_done:
            if preprocess_state is not None:
                state = preprocess_state(state)
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = np.clip(reward, -1, 1)
            is_done = done
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, [action], reward, done_mask))

            state = next_state
            score += reward
            counter += 1
            if done:
                break
        if counter > update_freq:
            agent.train()
            counter = 0

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, len(agent.buffer)))
        name = env_name.split("/")[-1]
        writer.add_scalar("%s_average_return" % name, score, episode)

    env.close()


def DQN_atari(env_name):
    # Generate environment
    env = gym.make(env_name, render_mode='rgb_array')

    # Params
    num_episode = 300
    update_freq = 1e3
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "replay_start_size": 5e3,
        "buffer_size": int(1e5),
        "target_update_freq": 1,
    }

    # Generate agent
    agent = DQN(configs)

    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, num_episode, update_freq, env_name, writer, process_atari)


if __name__ == '__main__':
    # DQN_atari("ALE/BankHeist-v5") # far below human's performance
    # DQN_atari("ALE/VideoPinball-v5") # much better than human's performance
    # DQN_atari("ALE/Freeway-v5") # close to human's performance
    DQN_atari("ALE/SpaceInvaders-v5") # better than human's performance
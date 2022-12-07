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
    env: gym.Env, agent:DQN, num_epoch: int,
    env_name: str, writer: SummaryWriter, preprocess_state = None
):

    scores = []
    scores_norm = []
    episode_cnt = 0
    step_cnt = 0
    name = env_name.split("/")[-1]
    train = True

    while train:
        score = 0
        score_norm = 0
        state = env.reset()
        done = False

        while not done:
            step_cnt += 1
            if preprocess_state is not None:
                state = preprocess_state(state)
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_norm = np.clip(reward, -1, 1)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, [action], reward, done_mask))
            state = next_state
            score += reward
            score_norm += reward_norm
            if step_cnt % 16 == 0:
                agent.train()
                if agent.learn_step_cnt % 50000 == 0:
                    epoch = agent.learn_step_cnt // 50000
                    writer.add_scalar("%s_average_return" % name, np.mean(scores), epoch)
                    writer.add_scalar("%s_average_return (normalized)" % name, np.mean(scores_norm), epoch)
                    if epoch == num_epoch:
                        train = False
        
        scores.append(score)
        scores_norm.append(score_norm)
        episode_cnt += 1
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode_cnt, score, len(agent.buffer)))

    env.close()


def DQN_atari(env_name):
    # Generate environment
    env = gym.make(env_name, render_mode='rgb_array')

    # Params
    num_episode = 300
    configs = {
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "replay_start_size": 5e4,
        "buffer_size": int(1e5),
        "target_update_freq": 1,
    }

    # Generate agent
    agent = DQN(configs)

    # Generate tensorboard writer
    writer = tensorboard_writer(env_name)
    train(env, agent, num_episode, env_name, writer, process_atari)


if __name__ == '__main__':
    # DQN_atari("ALE/BankHeist-v5") # far below human's performance
    # DQN_atari("VideoPinball-v4") # much better than human's performance
    # DQN_atari("ALE/Freeway-v5") # close to human's performance
    DQN_atari("SpaceInvaders-v4") # better than human's performance
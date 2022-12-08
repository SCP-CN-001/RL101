import time

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

num_episode = 300
# env_name = "VideoPinball-v4"
# env_name = "SpaceInvaders-v4"
env_name = "Pong-v4"
current_time = time.localtime()
timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)

env = gym.make(env_name, render_mode='rgb_array')
writer = SummaryWriter("./logs/random/%s/%s" % (env_name, timestamp))

scores = []

for i in range(num_episode):
    _ = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        score += reward
    scores.append(score)
    writer.add_scalar("%s_average_return" % env_name, np.mean(scores), i)

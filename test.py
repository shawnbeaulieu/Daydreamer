import gym
import atari_py

import numpy as np

env = gym.make('Frostbite-v0')
env.reset()
for _ in range(1000):
    env.render()
    print(env.action_space.n)
    action = np.random.choice(range(18))
    obs, reward, done, info = env.step(action)
    print(reward)

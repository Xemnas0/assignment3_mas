from __future__ import division
import gym
import numpy as np
from collections import deque
from gym import spaces


def create_env(env_id):
    env = gym.make(env_id)
    env = Normalize(env)
    return env


class Normalize(gym.Wrapper):
    def __init__(self, env):
        super(Normalize, self).__init__(env)
        self.frame = deque([], maxlen=1)
        self.obs_norm = MaxMin()

    def reset(self):
        ob = self.env.reset()
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        self.frame.append(ob)
        return self.observation()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        self.frame.append(ob)
        return self.observation(), rew, done, info

    def observation(self):
        assert len(self.frame) == 1
        return np.stack(self.frame, axis=0)


class MaxMin:
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs


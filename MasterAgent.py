import gym
import tensorflow as tf
import numpy as np
import os
from ActorCriticModel import ActorCriticModel


class MasterAgent:
    def __init__(self, args):
        self.game_name = 'BipedalWalker-v2'
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

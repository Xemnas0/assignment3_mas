import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from ActorCriticModel import ActorCriticModel
from RandomAgent import RandomAgent
from queue import Queue
import multiprocessing
from a3c_bipedalWalker import args
from Worker import Worker
import math


class MasterAgent:
    def __init__(self):
        self.game_name = args.env_name
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.opt = keras.optimizers.Adam(lr=args.lr)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()
        print(f'Number of workers: {multiprocessing.cpu_count()}')
        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                 '{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        # Normalize labels
        state[0] -= math.pi
        if state[8] < 0.5:
            state[8] = -1
        if state[13] < 0.5:
            state[13] = -1
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done and step_counter < 1600:
                env.render(mode='rgb_array')
                mu, sigma, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                cov_matrix = np.diag(sigma[0])
                action = tf.clip_by_value(np.random.multivariate_normal(mu[0], cov_matrix),
                                          clip_value_min=env.action_space.low,
                                          clip_value_max=env.action_space.high)
                # action = tf.clip_by_value(mu[0],
                #                           clip_value_min=env.action_space.low,
                #                           clip_value_max=env.action_space.high)

                state, reward, done, _ = env.step(action)
                # Normalize labels
                state[0] = (state[0] - math.pi) / math.pi
                if state[8] < 0.5:
                    state[8] = -1
                if state[13] < 0.5:
                    state[13] = -1
                print(math.fabs(state[4]-state[9]))
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

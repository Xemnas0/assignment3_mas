import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
											 'Bipedal Walker.')
parser.add_argument('--algorithm', default='a3c', type=str,
					help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
					help='Train our model.')
parser.add_argument('--lr', default=0.001,
					help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
					help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
					help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
					help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
					help='Directory in which you desire to save the model.')
args = parser.parse_args()


class RandomAgent:
	"""Random Agent that will play the specified game
	  Arguments:
		env_name: Name of the environment to be played
		max_eps: Maximum number of episodes to run agent for.
	"""
	def __init__(self, env_name, max_eps):
		self.env = gym.make(env_name)
		self.max_episodes = max_eps
		self.global_moving_average_reward = 0
		self.res_queue = Queue()

	def run(self):
		reward_avg = 0
		for episode in range(self.max_episodes):
			done = False
			self.env.reset()
			reward_sum = 0.0
			steps = 0
			while not done:
				# Sample randomly from the action space and step
				_, reward, done, _ = self.env.step(self.env.action_space.sample())
				steps += 1
				reward_sum += reward
			# Record statistics
			self.global_moving_average_reward = record(episode,
													   reward_sum,
													   0,
													   self.global_moving_average_reward,
													   self.res_queue, 0, steps)

			reward_avg += reward_sum
		final_avg = reward_avg / float(self.max_episodes)
		print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
		return final_avg

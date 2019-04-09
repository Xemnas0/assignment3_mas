import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


parser = argparse.ArgumentParser(description='Run A3C algorithm on an OpenAI gym game.')
parser.add_argument('--env_name', default='BipedalWalker-v2', type=str,
					help='Choose environment (default=\'BipedalWalker-v2\'.')
parser.add_argument('--algorithm', default='random', type=str,
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


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

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

if __name__ == '__main__':
  print(args)
  random = RandomAgent(args.env_name, args.max_eps)
  if args.train:
    random.train()
  else:
    random.run()
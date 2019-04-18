import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import os
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from MasterAgent import *
from RandomAgent import *
from ActorCriticModel import ActorCriticModel

parser = argparse.ArgumentParser(description='Run A3C algorithm on an OpenAI gym game.')
parser.add_argument('--env_name', default='BipedalWalker-v2', type=str,
                    help='Choose environment (default=\'BipedalWalker-v2\'.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=10, type=int,
                    help='How often to update the global model.')  # TODO: experiment with this
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--max_step_per_ep', default=200, type=int,
                    help='Maximum number of steps per episode.')
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
        f"Episode: {episode} | " +
        f"Moving Average Reward: {int(global_ep_reward)} | " +
        f"Episode Reward: {int(episode_reward)} | " +
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | " +
        f"Steps: {num_steps} | " +
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward


if __name__ == '__main__':
    print(args)

    # random = RandomAgent(args.env_name, args.max_eps)
    # if args.train:
    #     random.train()
    # else:
    #     random.run()
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()

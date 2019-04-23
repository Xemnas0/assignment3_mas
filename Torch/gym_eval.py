from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import gym
import logging


parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='BipedalWalker-v2',
    metavar='ENV',
    help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--new-gym-eval',
    default=False,
    metavar='NGE',
    help='Create a gym evaluation for upload')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

torch.manual_seed(args.seed)


d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = create_env("{}".format(args.env))
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3C_MLP(env.observation_space.shape[0], env.action_space)

if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, "{}_monitor".format(args.env), force=True)

player.model.load_state_dict(saved_state)

player.model.eval()
best_reward = 0
best_speed = 0
for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.eps_len = 0
    reward_sum = 0
    speed_sum = 0

    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        player.action_test()
        reward_sum += player.reward
        speed_sum += player.state.numpy()[0][2]
        if player.done:
            num_tests += 1
            if reward_sum > best_reward:
                best_reward = reward_sum
            if speed_sum > best_speed:
                best_speed = speed_sum
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "Episode_length, {0}, reward_sum, {1}, reward_mean, {2:.4f}, best_speed, {3}, best_reward, {4}".format(player.eps_len, reward_sum, reward_mean, best_speed, best_reward))
            break

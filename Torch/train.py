from __future__ import division
from setproctitle import setproctitle as ptitle

import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import gym


def train(rank, args, shared_model, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    torch.manual_seed(args.seed + rank)
    env = create_env(args.env)
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.model = A3C_MLP(
        player.env.observation_space.shape[0], player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()

    player.model.train()
    while True:

        player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            player.cx = Variable(torch.zeros(1, 128))
            player.hx = Variable(torch.zeros(1, 128))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
            
        for step in range(args.num_steps):

            player.action_train()

            if player.done:
                break

        if player.done:
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()

        else:
            R = torch.zeros(1, 1)
        if not player.done:
            state = player.state
            value, _, _, _ = player.model(
                (Variable(state), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (0.01 * player.entropies[i].sum())
        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()

import threading
from ActorCriticModel import ActorCriticModel
import gym
from Memory import Memory
import tensorflow as tf
import tensorflow_probability as tfp
from a3c_bipedalWalker import args, record
import numpy as np
import os
import math


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = -1e100
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 game_name=args.env_name,
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def run(self):
        total_step = 1
        mem = Memory()
        # Loop for all the episodes
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()

            obs = current_state.clip(self.mn_d, self.mx_d)
            current_state = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                        ) / (self.mx_d - self.mn_d)) + self.new_mind

            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            time_count = 1
            total_loss = tf.constant(10e5)
            # Loop through one episode, until done or reached maximum steps per episode
            for ep_t in range(args.max_step_per_ep):
                # Take action based on current state
                mu, sigma, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                cov_matrix = np.diag(sigma[0])
                normal_dist = tfp.distributions.Normal(mu, tf.sqrt(sigma))
                # action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                #                           clip_value_min=-0.999999,
                #                           clip_value_max=0.999999)

                action = tf.clip_by_value(mu,
                                          clip_value_min=self.env.action_space.low,
                                          clip_value_max=self.env.action_space.high)

                # Receive new state and reward
                # print(action.numpy()[0])
                new_state, reward, done_game, _ = self.env.step(action.numpy()[0])
                obs = new_state.clip(self.mn_d, self.mx_d)
                new_state = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                                  ) / (self.mx_d - self.mn_d)) + self.new_mind

                done = True if ep_t == args.max_step_per_ep - 1 else done_game

                reward = max(min(float(reward), 1.0), -10.0)
                ep_reward += reward

                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:

                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(total_loss)
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)

                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    try:
                        self.opt.apply_gradients(zip(grads,
                                                     self.global_model.trainable_weights))
                    except ValueError:
                        print("ValueError")



                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                        ep_steps += 1
                        time_count += 1
                        total_step += 1
                        break
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal TODO: Check why
        else:
            _, _, reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # print(discounted_rewards)

        # mu, sigma, values = self.local_model(
        #     tf.convert_to_tensor(np.vstack(memory.states),
        #                          dtype=tf.float32))
        # # Get our advantages
        # advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
        #                                  dtype=tf.float32) - values
        # # Critic loss
        # critic_loss = tf.square(advantage)
        #
        # # Actor loss
        # normal_dist = tfp.distributions.Normal(mu, sigma)
        # # print(normal_dist)
        # log_prob = tf.math.log(normal_dist.prob(np.array(memory.actions)) + 1e-10)
        #
        # # Entropy
        # entropy = normal_dist.entropy()  # encourage exploration
        #
        # actor_loss = - log_prob * tf.stop_gradient(advantage) - 0.01 * entropy
        # # actor_loss = - log_prob * advantage

        actor_loss = 0
        critic_loss = 0
        gae = 0
        for i in range(len(memory.states)):
            mu, sigma, values = self.local_model(tf.convert_to_tensor(memory.states[i][None, :], dtype=tf.float32))
            try:
                _, _, next_value = self.local_model(tf.convert_to_tensor(memory.states[i+1][None, :], dtype=tf.float32))
            except:
                _, _, next_value = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))
            advantage = tf.convert_to_tensor(np.array(discounted_rewards[i]),
                                         dtype=tf.float32) - values
            critic_loss = critic_loss + tf.square(advantage)

            normal_dist = tfp.distributions.Normal(mu, sigma)
            log_prob = tf.math.log(normal_dist.prob(np.array(memory.actions[i])) + 1e-10)
            entropy = normal_dist.entropy()

            # Generalized Advantage Estimation
            delta_t = memory.rewards[i] + gamma * next_value - values
            #
            gae = gae * gamma + delta_t

            actor_loss = actor_loss - tf.reduce_mean(log_prob) * gae - 0.001 * tf.reduce_mean(entropy)

        # print("sigma: ", sigma[0])
        # print("mu: ", mu[0])
        # print("val: ", values)
        # print("dis rew: ", discounted_rewards)
        # print("crit: ", tf.reduce_mean(critic_loss))
        # print("actor: ", tf.reduce_mean(actor_loss))
        # print("entropy: ", tf.reduce_mean(entropy))
        total_loss = tf.reduce_mean(0.5 * critic_loss) + actor_loss
        # print("loss: ", total_loss)
        return total_loss

import threading
from ActorCriticModel import ActorCriticModel
import gym
from Memory import Memory
import tensorflow as tf
import tensorflow_probability as tfp
from a3c_bipedalWalker import args, record
import numpy as np
import os


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

    def run(self):
        total_step = 1
        mem = Memory()
        # Loop for all the episodes
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            time_count = 1
            # Loop through one episode, until done or reached maximum steps per episode
            for ep_t in range(args.max_step_per_ep):
                mu, sigma, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                cov_matrix = np.diag(sigma[0])
                action = tf.clip_by_value(np.random.multivariate_normal(mu[0], cov_matrix),
                                          clip_value_min=self.env.action_space.low,
                                          clip_value_max=self.env.action_space.high)
                new_state, reward, done_game, _ = self.env.step(action)

                done = True if ep_t == args.max_step_per_ep - 1 else False
                if done_game:
                    ep_reward += 100
                ep_reward += reward
                # reward = (reward + 8) / 8
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_variables)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_variables))
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
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # print(discounted_rewards)

        mu, sigma, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Critic loss
        critic_loss = tf.reduce_mean(0.5*tf.square(advantage))

        # Actor loss
        normal_dist = tfp.distributions.Normal(mu, sigma + 1e-16)
        log_prob = normal_dist.log_prob(np.array(memory.actions) + 1e-10)
        entropy = normal_dist.entropy()
        exp_v = log_prob * tf.stop_gradient(advantage) - 1e-2 * entropy
        actor_loss = tf.reduce_mean(exp_v)

        # log_prob = normal_dist.log_prob(self.a_his)
        # exp_v = log_prob * tf.stop_gradient(td)
        # entropy = normal_dist.entropy()  # encourage exploration
        # self.exp_v = ENTROPY_BETA * entropy + exp_v
        # self.a_loss = tf.reduce_mean(-self.exp_v)
        # print(entropy)
        # print(critic_loss, ", ", actor_loss)
        total_loss = critic_loss + actor_loss
        return total_loss

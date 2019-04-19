import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        w_init = keras.initializers.normal(0, 0.1)
        w_uniform_init = keras.initializers.RandomUniform(minval=-0.99, maxval=0.99, seed=None)
        self.dense1 = layers.Dense(200, activation=tf.nn.relu6, kernel_initializer=w_init)
        self.actions_mean = layers.Dense(action_size, activation=tf.nn.tanh, kernel_initializer=w_uniform_init)
        self.actions_sigma = layers.Dense(action_size, activation=tf.nn.softplus, kernel_initializer=w_init)

        self.dense2 = layers.Dense(100, activation=tf.nn.relu6, kernel_initializer=w_init)
        self.values = layers.Dense(1, activation=None, kernel_initializer=w_init)

    def call(self, inputs):
        """
        Forward pass
        """

        # Actor
        x = self.dense1(inputs)
        mu = self.actions_mean(x)
        sigma = self.actions_sigma(x) + 0.001

        # Critic
        v1 = self.dense1(inputs)
        values = self.values(v1)

        return mu, sigma, values

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        # self.action_size = action_size
        self.action_size = 14
        w_init = keras.initializers.normal(0., 0.1)
        w_uniform_init = keras.initializers.RandomUniform(minval=-0.99, maxval=0.99, seed=None)

        # Conv model
        # self.conv1 = layers.Conv1D(32, 3, strides=1, padding="same")
        # self.conv2 = layers.Conv1D(32, 3, strides=1, padding="same")
        # self.conv3 = layers.Conv1D(64, 2, strides=1, padding="same")
        # self.conv4 = layers.Conv1D(64, 1, strides=1)

        # MLP model
        self.fc1 = layers.Dense(256, kernel_initializer=w_init)
        self.fc2 = layers.Dense(256, kernel_initializer=w_init)
        self.fc3 = layers.Dense(128, kernel_initializer=w_init)
        self.fc4 = layers.Dense(128, kernel_initializer=w_init)


        # self.lstm = layers.LSTM(128, input_shape=(1, 128))
        # self.state = self.lstm.get_initial_state(self.action_size, dtype=tf.float32)

        # self.dense1 = layers.Dense(200, activation=tf.nn.relu6)
        self.actions_mean = layers.Dense(action_size, activation=tf.nn.softsign, kernel_initializer=w_init)
        self.actions_sigma = layers.Dense(action_size, activation=tf.nn.softplus, kernel_initializer=w_init)

        # self.dense2 = layers.Dense(100, activation=tf.nn.relu6)
        self.values = layers.Dense(1, kernel_initializer=w_init)

    def call(self, inputs):
        """
        Forward pass
        """

        # Actor
        x = self.fc1(inputs)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = self.fc2(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = self.fc3(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = self.fc4(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = self.lstm(x)

        # x = self.dense1(x)
        mu = self.actions_mean(x)
        sigma = self.actions_sigma(x) + 0.001

        # Critic
        # v1 = self.dense2(inputs)
        values = self.values(x)

        return mu, sigma, values

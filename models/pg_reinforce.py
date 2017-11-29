import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np


class PolicyGradientAgent(object):

    """This is an implementation of an RL agent using the REINFORCE Policy Gradient algorithm"""
    def __init__(self, sess: tf.Session, discount=0.99, board_size=(2, 7), num_actions=7, name='agent_name'):
        self.sess = sess
        self.discount = discount
        self.board_size = board_size
        self.num_actions = num_actions
        self.name = name

        # Record reward history for normalization
        self.all_rewards = []
        self.max_reward_length = 1000000

        # Rollout data.
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

        self.build_model()

    def build_model(self):
        self.board = tf.placeholder(tf.float32, shape=[None, self.board_size[0], self.board_size[1]],
                                    name='board')
        self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, self.num_actions],
                                                name='binary_mask')
        self.logits = self.policy_network(is_training=True, reuse=False)
        # Sample an action from the policy distribution and make it a scalar.
        self.sample = tf.reshape(tf.multinomial(self.logits, 1), [])
        self.actions = tf.placeholder(tf.int32)
        self.discounted_rewards = tf.placeholder(tf.int32)

        log_prob = tf.log(tf.nn.softmax(self.logits))

        # Get the log probabilities of the actions
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # Define optimization procedure
        loss = -tf.reduce_mean(tf.multiply(act_prob, self.discounted_rewards))
        optimiser = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        self.train = optimiser.minimize(loss)

    def policy_network(self, is_training=True, reuse=False):
        w_init = tfl.xavier_initializer()

        with tf.variable_scope('policy_network/'.format({self.name}), reuse=reuse):
            h0 = tfl.fully_connected(tf.float32, inputs=self.board, num_outputs=20, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=w_init)
            h1 = tfl.fully_connected(tf.float32, inputs=h0, num_outputs=20, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=w_init)
            all_actions_logits = tfl.fully_connected(tf.float32, inputs=h1, num_outputs=self.num_actions,
                                                     weights_initializer=w_init)
            logits = tf.multiply(all_actions_logits, self.valid_action_mask)
            return logits

    def sample_action(self, board, valid_action_mask):
        feed_dict = {
            self.board: board,
            self.valid_action_mask: valid_action_mask,
        }
        return self.sess.run(self.sample, feed_dict=feed_dict)

    def run_train_step(self):
        feed_dict = {
            self.board: self.states,
            self.actions: self.actions,
            self.discounted_rewards: self.discounted_rewards(self.rewards),
            self.valid_action_mask: self.masks,
        }
        self.sess.run(self.train, feed_dict=feed_dict)
        self.clean_up_rollout()

    def store_rollout(self, state, action, reward, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)

    def clean_up_rollout(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def discounted_rewards(self, rewards) -> np.array:
        n = len(rewards)
        r = 0
        discounted_rewards = np.zeros(n)

        for t in reversed(range(n)):
            r = rewards[t] + self.discount * r
            discounted_rewards[t] = r

        self.all_rewards += discounted_rewards.tolist()
        self.all_rewards = self.all_rewards[-self.max_reward_length:]
        discounted_rewards -= np.mean(self.all_rewards)
        discounted_rewards /= np.std(self.all_rewards)

        return discounted_rewards




import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np


class PolicyGradientAgent(object):

    """This is an implementation of an RL agent using the REINFORCE Policy Gradient algorithm"""
    def __init__(self, sess: tf.Session, discount=0.99, board_size=(2, 8), num_actions=8, name='agent_name'):
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

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.build_model()
        self.configure_training_procedure()

        tf.global_variables_initializer().run()

    def build_model(self):
        self.board = tf.placeholder(tf.float32, shape=[None, self.board_size[0], self.board_size[1], 1],
                                    name='board')
        self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, self.num_actions],
                                                name='binary_mask')
        self.taken_actions = tf.placeholder(tf.int32, shape=[None], name='taken_actions')
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name='discounted_rewards')

        # Outputs of the policy network
        self.logits = self.policy_network(is_training=True, reuse=False)
        self.sample = tf.reshape(tf.multinomial(self.logits, 1), [])
        self.action_prob = tf.nn.softmax(self.logits)

    def configure_training_procedure(self):
        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(self.logits)[0]) * tf.shape(self.logits)[1] + self.taken_actions
        self.picked_act_logits = tf.gather(tf.reshape(self.logits, [-1]), indices)

        self.loss = -tf.reduce_sum(tf.multiply(self.picked_act_logits, self.discounted_rewards))

        # compute gradients
        self.optimiser = tf.train.RMSPropOptimizer(0.0002)
        self.train_op = self.optimiser.minimize(self.loss)

    def policy_network(self, is_training=True, reuse=False):
        w_init = tfl.xavier_initializer()

        with tf.variable_scope('policy_network/{}'.format(self.name), reuse=reuse):
            net_h0 = tfl.conv2d(inputs=self.board, num_outputs=32, kernel_size=2, stride=1, padding='SAME',
                                activation_fn=tf.nn.relu, weights_initializer=w_init)
            net_h1 = tfl.conv2d(inputs=net_h0, num_outputs=32, kernel_size=2, padding='SAME', stride=1,
                                activation_fn=tf.nn.relu, weights_initializer=w_init)
            net_h2 = tfl.conv2d(inputs=net_h1, num_outputs=1, kernel_size=2, padding='VALID', stride=1,
                                activation_fn=tf.nn.relu, weights_initializer=w_init)
            net_h2 = tf.reshape(net_h2, shape=[-1, 7])
            all_actions_logits = tfl.fully_connected(inputs=net_h2, num_outputs=self.num_actions, activation_fn=None,
                                                     weights_initializer=w_init)
            return all_actions_logits - self.valid_action_mask

    def sample_action(self, board, valid_action_mask):
        feed_dict = {
            self.board: board[np.newaxis, :],
            self.valid_action_mask: valid_action_mask[np.newaxis, :],
        }
        a, b = self.sess.run([self.sample, self.action_prob], feed_dict=feed_dict)
        return a

    def run_train_step(self):
        feed_dict = {
            self.board: self.states,
            self.taken_actions: self.actions,
            self.discounted_rewards: self.compute_discounted_rewards(self.rewards),
            self.valid_action_mask: self.masks,
        }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.clean_up_rollout()

        return loss

    def get_average_reward(self):
        return np.mean(self.all_rewards)

    def store_rollout(self, state, action, reward, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)

    def clean_up_rollout(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def compute_discounted_rewards(self, rewards) -> np.array:
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




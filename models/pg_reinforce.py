import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
import os
from models.ops import batch_normalization


class PolicyGradientAgent(object):

    """This is an implementation of an RL agent using the REINFORCE Policy Gradient algorithm"""
    def __init__(self, sess: tf.Session, discount=0.99, board_size=(2, 8), num_actions=8, reuse=False, is_training=True,
                 name='agent_name', checkpoint_dir='models/checkpoints/'):
        self.sess = sess
        self.discount = discount
        self.board_size = board_size
        self.num_actions = num_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.reuse = reuse
        self.is_training = is_training

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
        _, self.logits = self.policy_network(is_training=self.is_training, reuse=self.reuse)
        self.action_prob, _ = self.policy_network(is_training=False, reuse=True)

    def configure_training_procedure(self):
        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(self.logits)[0]) * tf.shape(self.logits)[1] + self.taken_actions
        self.picked_act_logits = tf.gather(tf.reshape(self.logits, [-1]), indices)

        self.loss = -tf.reduce_mean(tf.multiply(self.picked_act_logits, self.discounted_rewards))

        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars if 'policy_network_{}'.format(self.name) in var.name]

        self.saver = tf.train.Saver()

        # compute gradients
        self.optimiser = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
        self.train_op = self.optimiser.minimize(self.loss, var_list=self.vars)

    def policy_network(self, is_training=True, reuse=False):
        w_init = tfl.xavier_initializer()
        gamma_init = tf.random_normal_initializer(1., 0.02)
        epsilon = 1e-100

        with tf.variable_scope('policy_network_{}'.format(self.name), reuse=reuse):
            net_h0 = tf.layers.conv2d(inputs=self.board, filters=8, kernel_size=2, strides=1, padding='SAME',
                                      activation=tf.nn.relu, kernel_initializer=w_init, name='pg_h0/conv2d')
            net_h1 = tf.layers.conv2d(inputs=net_h0, filters=16, kernel_size=2, padding='SAME', strides=1,
                                      activation=None, kernel_initializer=w_init, name='pg_h1/conv2d')
            net_h1 = batch_normalization(net_h1, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h1/batch_norm')
            net_h2 = tf.layers.conv2d(inputs=net_h1, filters=32, kernel_size=2, padding='SAME', strides=1,
                                      activation=None, kernel_initializer=w_init, name='pg_h2/conv2d')
            net_h2 = batch_normalization(net_h2, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h2/batch_norm')

            # Residual layer
            net = tf.layers.conv2d(inputs=net_h2, filters=8, kernel_size=2, strides=(1, 1),
                                   padding='SAME', activation=None, kernel_initializer=w_init,
                                   name='pg_h3_res/conv2d')
            net = batch_normalization(net, is_training=is_training, initializer=gamma_init,
                                      activation=tf.nn.relu, name='pg_h3_res/batch_norm')
            net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=2, strides=(1, 1),
                                   padding='SAME', activation=tf.nn.relu, kernel_initializer=w_init,
                                   name='pg_h3_res/conv2d2')
            net = batch_normalization(net, is_training=is_training, initializer=gamma_init,
                                      activation=tf.nn.relu, name='pg_h3_res/batch_norm2')
            net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=2, strides=(1, 1),
                                   padding='SAME', activation=None, kernel_initializer=w_init,
                                   name='pg_h3_res/conv2d3')
            net = batch_normalization(net, is_training=is_training, initializer=gamma_init,
                                      activation=tf.nn.relu, name='pg_h3_res/batch_norm3')
            net_h3 = tf.add(net_h2, net, name='pg_h3/add')
            net_h3 = tf.nn.relu(net_h3, name='pg_h3/add_lrelu')

            net_h4 = tf.layers.conv2d(inputs=net_h3, filters=1, kernel_size=2, padding='VALID', strides=1,
                                      activation=None, kernel_initializer=w_init, name='pg_h4/cov2d')
            net_h4 = batch_normalization(net_h4, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h4/batch_norm')
            net_h4 = tf.reshape(net_h4, shape=[-1, 7], name='pg_h4/reshaped')

            logits = tf.layers.dense(inputs=net_h4, units=self.num_actions, activation=None,
                                     kernel_initializer=w_init, name='pg_logits')
            exp_logits = tf.exp(logits, name='pg_exp_logits')
            action_prob = (exp_logits * self.valid_action_mask) \
                / (tf.reduce_sum(exp_logits * self.valid_action_mask) + epsilon)

            return action_prob, logits

    def sample_action(self, board, valid_action_mask):
        feed_dict = {
            self.board: board[np.newaxis, :],
            self.valid_action_mask: valid_action_mask[np.newaxis, :],
        }
        action_prob = self.sess.run(self.action_prob, feed_dict=feed_dict)
        action_prob = np.ndarray.flatten(action_prob)
        return np.random.choice(self.num_actions, p=action_prob)

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

    def save_model_params(self, file, step):
        path = os.path.join(self.checkpoint_dir, file)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, path)

    def restore_model_params(self, file, step):
        path = os.path.join(self.checkpoint_dir, file)
        self.saver.restore(self.sess, path)

    def transfer_params(self, other_agent):
        transfer_op = [other_var.assign(this_var.value()) for other_var, this_var in zip(other_agent.vars, self.vars)]
        self.sess.run(transfer_op)


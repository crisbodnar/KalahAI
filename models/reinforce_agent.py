import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
import os
from models.ops import batch_normalization


class PolicyGradientAgent(object):

    """This is an implementation of an RL agent using the REINFORCE Policy Gradient algorithm"""
    def __init__(self, sess: tf.Session, discount=0.85, board_size=(2, 8), num_actions=8, reuse=False, is_training=True,
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
        self.define_summaries()
        tf.global_variables_initializer().run()

    def build_model(self):
        self.board = tf.placeholder(tf.float32, shape=[None, self.board_size[0], self.board_size[1], 1],
                                    name='board')
        self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, self.num_actions],
                                                name='binary_mask')
        self.taken_actions = tf.placeholder(tf.int32, shape=[None], name='taken_actions')
        self.discounted_rewards = tf.placeholder(tf.float32, shape=[None], name='discounted_rewards')

        # Outputs of the policy network
        self.action_prob, self.logits = self.policy_network(is_training=False, reuse=self.reuse)
        self.action_prob_sampler, _ = self.policy_network(is_training=False, reuse=True)

    def configure_training_procedure(self):
        # Get the probability of the selected actions in te episode
        picked_action_mask = tf.one_hot(self.taken_actions, self.num_actions, 1.0, 0.0)
        self.picked_action_prob = tf.reduce_sum(self.action_prob * picked_action_mask, 1)

        # Get all the weights to be trained
        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars if 'policy_network_{}'.format(self.name) in var.name]

        # Define the losses
        self.pg_loss = tf.reduce_mean(-tf.log(self.picked_action_prob) * self.discounted_rewards)
        # self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.vars])
        self.loss = self.pg_loss  # + 0.002 * self.reg_loss

        # Define the optimiser to compute the gradients
        self.optimiser = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
        self.train_op = self.optimiser.minimize(self.loss, var_list=self.vars)

        # Create a saver to backup the weights during training
        self.saver = tf.train.Saver()

    def define_summaries(self):
        self.loss_sum = tf.summary.scalar('loss_sum', self.loss)
        self.action_prob_sum = tf.summary.tensor_summary('action_prob_sum', self.action_prob)
        self.logits_sum = tf.summary.tensor_summary('logits_sum', self.logits)
        self.out_sum = tf.summary.merge([self.action_prob_sum, self.logits_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def policy_network(self, is_training=True, reuse=False):
        w_init = tfl.xavier_initializer(uniform=True)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope('policy_network_{}'.format(self.name), reuse=reuse):
            flattened_imp = tf.contrib.layers.flatten(self.board)
            net_h1 = tf.layers.dense(inputs=flattened_imp, units=10, activation=None,
                                     kernel_initializer=w_init, name='pg_h1')
            net_h1 = batch_normalization(net_h1, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h1/batch_norm')

            net_h2 = tf.layers.dense(inputs=net_h1, units=10, activation=None,
                                     kernel_initializer=w_init, name='pg_h2')
            net_h2 = batch_normalization(net_h2, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h2/batch_norm')

            net_h3 = tf.layers.dense(inputs=net_h2, units=10, activation=None,
                                     kernel_initializer=w_init, name='pg_h3')
            net_h3 = batch_normalization(net_h3, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h3/batch_norm')

            net_h4 = tf.layers.dense(inputs=net_h3, units=10, activation=None,
                                     kernel_initializer=w_init, name='pg_h4')
            net_h4 = batch_normalization(net_h4, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h4/batch_norm')

            net_h5 = tf.layers.dense(inputs=net_h4, units=10, activation=None,
                                     kernel_initializer=w_init, name='pg_h5')
            net_h5 = batch_normalization(net_h5, is_training=is_training, initializer=gamma_init,
                                         activation=tf.nn.relu, name='pg_h5/batch_norm')

            logits = tf.layers.dense(inputs=net_h5, units=self.num_actions, activation=None,
                                     kernel_initializer=w_init, name='pg_logits')

            # Make the logits numerically stable for computing the softmax
            # stable_logits = tf.identity(logits - tf.reduce_max(logits, axis=0), name='pg_stable_logits')

            # Compute the unnormalised probabilities
            exp_logits = tf.exp(logits, name='pg_unnormal_prob')
            valid_exp_logits = exp_logits * self.valid_action_mask

            # Compute probabilities taking into account only the valid actions
            action_prob = valid_exp_logits / tf.reduce_sum(valid_exp_logits)

            return action_prob, logits

    def sample_action(self, board, valid_action_mask):
        feed_dict = {
            self.board: board[np.newaxis, :],
            self.valid_action_mask: valid_action_mask[np.newaxis, :],
        }
        action_prob, logits, out_sum = self.sess.run([self.action_prob_sampler, self.logits, self.out_sum],
                                                     feed_dict=feed_dict)
        action_prob = np.ndarray.flatten(action_prob)

        self.writer.add_summary(out_sum)
        # print(board)
        # print(valid_action_mask)
        # print(logits)
        # print(action_prob)

        return np.random.choice(self.num_actions, p=action_prob)

    def get_best_action(self, board, valid_action_mask):
        feed_dict = {
            self.board: board[np.newaxis, :],
            self.valid_action_mask: valid_action_mask[np.newaxis, :],
        }
        action_prob = self.sess.run(self.action_prob, feed_dict=feed_dict)
        action_prob = np.ndarray.flatten(action_prob)
        return np.argmax(action_prob)

    def run_train_step(self, counter):
        feed_dict = {
            self.board: self.states,
            self.taken_actions: self.actions,
            self.discounted_rewards: self.compute_discounted_rewards(self.rewards),
            self.valid_action_mask: self.masks,
        }
        _, loss, loss_sum = self.sess.run([self.train_op, self.loss, self.loss_sum], feed_dict=feed_dict)
        self.clean_up_rollout()
        self.writer.add_summary(loss_sum, counter)
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

    def save_model_params(self, file):
        path = os.path.join(self.checkpoint_dir, file)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, path)

    def restore_model_params(self, file):
        path = os.path.join(self.checkpoint_dir, file)
        self.saver.restore(self.sess, path)

    def transfer_params(self, other_agent):
        transfer_op = [other_var.assign(this_var.value()) for other_var, this_var in zip(other_agent.vars, self.vars)]
        self.sess.run(transfer_op)


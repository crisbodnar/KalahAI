import tensorflow as tf
import tensorflow.contrib.layers as tfl


class PolicyGradientAgent(object):
    """This is an implementation of an RL agent using the REINFORCE Policy Gradients with Advantage algorithm"""

    def __init__(self, sess: tf.Session, discount=0.99, board_size=(2, 7), num_actions=7, batch_size=1):
        self.sess = sess
        self.discount = discount
        self.board_size = board_size
        self.num_actions = num_actions
        self.batch_size = batch_size

    def build_model(self):
        self.board = tf.placeholder(tf.float32, shape=[self.batch_size, self.board_size[0], self.board_size[1]],
                                    name='board')
        self.valid_action_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_actions],
                                                name='binary_mask')

    def policy_network(self, board, valid_action_mask, is_training=True, reuse=False):
        w_init = tfl.xavier_initializer()

        with tf.variable_scope('policy_network', reuse=reuse):
            h0 = tfl.fully_connected(tf.float32, inputs=board, num_outputs=20, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=w_init)
            h1 = tfl.fully_connected(tf.float32, inputs=h0, num_outputs=20, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=w_init)
            all_actions_logits = tfl.fully_connected(tf.float32, inputs=h1, num_outputs=self.num_actions,
                                                     weights_initializer=w_init)
            v

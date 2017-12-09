import tensorflow as tf
import numpy as np


class ACNetwork(object):
    def __init__(self, state_shape: [int], num_act: int):
        w_init = tf.contrib.layers.xavier_initializer()

        # Base network
        self.state = tf.placeholder(tf.float32, shape=[None] + state_shape)
        self.mask = tf.placeholder(tf.float32, shape=[None, num_act])
        inverse_mask = tf.ones_like(self.mask) - self.mask

        flattened_imp = tf.contrib.layers.flatten(self.state)
        net_h1 = tf.layers.dense(inputs=flattened_imp, units=16, activation=tf.nn.relu, kernel_initializer=w_init)
        net_h2 = tf.layers.dense(inputs=net_h1, units=20, activation=tf.nn.relu, kernel_initializer=w_init)
        net_h3 = tf.layers.dense(inputs=net_h2, units=10, activation=tf.nn.relu, kernel_initializer=w_init)

        # Policy network
        self.logits = tf.layers.dense(net_h3, num_act, activation=None, kernel_initializer=w_init)
        # Zero the probabilities of invalid actions
        self.logits = self.logits * self.mask - inverse_mask * 1e35

        # Value network
        self.value = tf.layers.dense(net_h3, 1, activation=None, kernel_initializer=w_init)

        # The variables of this network
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def sample(self, state: np.array, mask: np.array) -> (int, int):
        """Renormalises the logits by taking into account only vthe valid actions and generates a sample"""
        sess = tf.get_default_session()

        feed_dict = {
            self.state: [state],
            self.mask: [mask],
        }

        logits, value = sess.run([self.logits, self.value], feed_dict)

        # Sampling is done in numpy because it has higher precision and it is less likely to have numerical problems
        logits = np.asarray(logits, dtype=np.float64).flatten()
        # The max logit is subtracted for numerical stability. This uses the property softmax(x - a) = softmax(x)
        exp = np.exp(logits - np.max(logits))
        dist = exp / np.sum(exp)

        if np.sum(exp) == 0:
            raise ZeroDivisionError('There is no valid action. This should not happen')
        return np.random.choice(range(logits.size), p=dist), value[0][0]

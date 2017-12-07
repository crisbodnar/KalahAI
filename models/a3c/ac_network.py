import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from models.a3c.helpers import normalized_columns_initializer


class ActorCriticNetwork(object):
    def __init__(self, a_size, scope, trainer):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(tf.float32, shape=[None, 2, 8, 1], name='board')
            # self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, a_size], name='binary_mask')
            # inverse_mask = tf.subtract(tf.ones_like(self.valid_action_mask) - self.valid_action_mask)

            flattened_imp = tf.contrib.layers.flatten(self.inputs)
            net_h1 = tf.layers.dense(inputs=flattened_imp, units=16, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h1')
            net_h2 = tf.layers.dense(inputs=net_h1, units=20, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h2')
            net_h3 = tf.layers.dense(inputs=net_h2, units=10, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h3')

            # Output layers for policy and value estimations
            self.logits = slim.fully_connected(net_h3, a_size, activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(0.01),
                                          biases_initializer=None)
            # Compute probabilities taking into account only the valid actions
            self.policy = tf.nn.softmax(self.logits)
            # valid_prob = tf.add(tf.multiply(softmax, self.valid_action_mask), tf.multiply(inverse_mask, 1e-35))
            # self.policy = tf.divide(valid_prob, tf.reduce_sum(valid_prob, axis=1, keep_dims=True))

            self.value = slim.fully_connected(net_h3, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

                # Loss functions
                eps = 1e-7
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + eps))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + eps) * self.advantages)
                # self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in local_vars])
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01  # + 0.002 * self.reg_loss

                # Get gradients from local network using local losses
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 100.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

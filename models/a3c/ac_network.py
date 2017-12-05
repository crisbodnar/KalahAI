import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from models.a3c.helpers import normalized_columns_initializer


class ActorCriticNetwork(object):
    def __init__(self, s_size, a_size, scope, trainer):
        w_init = tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(tf.float32, shape=[None, 2, 8, 1], name='board')
            self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, a_size], name='binary_mask')

            flattened_imp = tf.contrib.layers.flatten(self.inputs)
            net_h1 = tf.layers.dense(inputs=flattened_imp, units=20, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h1')
            net_h2 = tf.layers.dense(inputs=net_h1, units=10, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h2')
            net_h3 = tf.layers.dense(inputs=net_h2, units=10, activation=tf.nn.relu,
                                     kernel_initializer=w_init, name='pg_h3')

            # Output layers for policy and value estimations
            logits = slim.fully_connected(net_h3, a_size, activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(0.01),
                                          biases_initializer=None, name='pg_logits')
            # Compute the unnormalised probabilities
            exp_logits = tf.exp(logits, name='pg_unnormal_prob')
            valid_exp_logits = exp_logits * self.valid_action_mask

            # Compute probabilities taking into account only the valid actions
            self.valid_action_prob = valid_exp_logits / tf.reduce_sum(valid_exp_logits)
            self.policy = tf.nn.softmax(logits)

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

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
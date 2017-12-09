import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import logging
from magent.mancala import MancalaEnv
from magent.move import Move
from magent.side import Side
from models.a3c.helpers import normalized_columns_initializer

logging.basicConfig(filename='a3c.client.log', level=logging.DEBUG)


class MLClient(object):
    """This is a client to access A3C models and make decision on them"""

    def __init__(self, sess: tf.Session, num_actions=7, name='agent_name', checkpoint_dir='./checkpoints/a3c'):
        self.sess = sess
        self.num_actions = num_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir

        self.build_model(num_actions)

        saver = tf.train.Saver(max_to_keep=5)
        logging.debug('Loading Model...')
        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        except Exception as e:
            logging.error("Failed to restore models", str(e))

    def build_model(self, a_size):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("global"):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(tf.float32, shape=[None, 2, 8, 1], name='board')
            self.valid_action_mask = tf.placeholder(tf.float32, shape=[None, a_size], name='binary_mask')

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
            softmax = tf.nn.softmax(self.logits)
            valid_prob = softmax * self.valid_action_mask
            self.policy = valid_prob / tf.reduce_sum(valid_prob, axis=1, keep_dims=True)

            self.value = slim.fully_connected(net_h3, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

    def get_best_action(self, env: MancalaEnv):
        flip_board = env.side_to_move == Side.NORTH
        state = env.board.get_board_image(flipped=flip_board)

        a_dist, v, logits = self.sess.run(
            [self.policy, self.value, self.logits],
            feed_dict={self.inputs: [state],
                       self.valid_action_mask: [env.get_action_mask_with_no_pie()],
                       }
        )

        action_prob = np.ndarray.flatten(a_dist)
        return np.argmax(action_prob) + 1


def main(_):
    with tf.Session() as sess:
        client = MLClient(sess)
        env = MancalaEnv()
        while not env.is_game_over():
            move = int(client.get_best_action(env))
            env.perform_move(Move(env.side_to_move, move))
            print(move)

        # pg_trainer = PolicyGradientTrainer(agent, env)
        # pg_trainer.train()


if __name__ == '__main__':
    tf.app.run()

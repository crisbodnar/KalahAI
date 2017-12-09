import numpy as np
import tensorflow as tf
import scipy.signal

from collections import namedtuple

# A batch which contains all the states, actions, advantages and discounted rewards of a rollout
Batch = namedtuple("Batch", ["states", "actions", "advantages", "discounted_rewards", "masks"])


class FastSaver(tf.train.Saver):
    """Disables the write meta graph argument to speed up the saver"""

    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=False, write_state=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, write_meta_graph, write_state)


def generate_training_batch(rollout, gamma: float, bootstrap_value=0.0):
    """Computes the advantages and prepares the batch for training

       The bootstrap value is used only for incomplete episodes which is not the case in our case where we always
       play a full Mancala Game. We can think of the environment's MDP final state as a loop state
       which always produces a reward of 0. This justifies the default value of 0 which we always use.
    """
    states = np.asarray(rollout.states)
    actions = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    values = np.asarray(rollout.values)
    masks = np.asarray(rollout.masks)

    # The advantage function is "Generalized Advantage Estimation"
    # For more details: https://arxiv.org/abs/1506.02438
    rewards_plus = np.concatenate((rewards.tolist(), [bootstrap_value]))
    discounted_rewards = discount(rewards_plus[:-1], gamma)
    value_plus = np.concatenate((values.tolist(), [bootstrap_value]))
    advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
    advantages = discount(advantages, gamma)

    return Batch(states, actions, advantages, discounted_rewards, masks)


def discount(x, gamma):
    """Calculates discounted rewards"""
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class Rollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.win = 0

    def add(self, state: np.array, action: int, reward: int, value: int, mask: [float]):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)

    def update_last_reward(self, reward):
        assert len(self.rewards) > 0
        self.rewards[-1] = reward

    def add_win(self):
        self.win = 1

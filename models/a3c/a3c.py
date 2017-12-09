import tensorflow as tf
from magent.mancala import MancalaEnv
from magent.side import Side
from magent.move import Move
import random
from models.a3c.helpers import Rollout, generate_training_batch
from models.a3c.model import ACNetwork
from models.a3c.agent import Agent
import numpy as np


class EnvironmentRunner(object):
    def __init__(self, env: MancalaEnv, ac_net: ACNetwork):
        self.env = env
        self.ac_net = ac_net
        self.sess = None
        self.opp_agent = None
        self.trainer_side = None

    def run(self, sess: tf.Session, opp_agent: Agent) -> Rollout:
        self.sess = sess
        self.opp_agent = opp_agent
        with self.sess.as_default():
            return self._run()

    def _run(self) -> Rollout:
        # Choose randomly the side to play
        self.trainer_side = Side.SOUTH if random.randint(0, 1) == 0 else Side.NORTH
        # Reset the environment so everything is in a clean state.
        self.env.reset()

        rollout = Rollout()
        while not self.env.is_game_over():
            # There is no choice if only one action is left. Taking that action automatically must be seen as
            # a characteristic behaviour of the environment. This helped the learning of the agent
            # to be more numerically stable (this is an empirical observation).
            if len(self.env.get_legal_moves()) == 1:
                action_left_to_perform = self.env.get_legal_moves()[0]
                self.env.perform_move(action_left_to_perform)
                continue

            if self.env.side_to_move == self.trainer_side:
                # If the agent is playing as NORTH, it's input would be a flipped board
                flip_board = self.env.side_to_move == Side.NORTH
                state = self.env.board.get_board_image(flipped=flip_board)
                mask = self.env.get_action_mask_with_no_pie()

                action, value = self.ac_net.sample(state, mask)
                # Because the pie move with index 0 is ignored, the action indexes must be shifted by one
                reward = self.env.perform_move(Move(self.trainer_side, action + 1))
                rollout.add(state, action, reward, value, mask)
            else:
                assert self.env.side_to_move == Side.opposite(self.trainer_side)
                action = self.opp_agent.produce_action(self.env.board.get_board_image(),
                                                       self.env.get_action_mask_with_no_pie(),
                                                       self.env.side_to_move)
                self.env.perform_move(Move(self.env.side_to_move, action + 1))

        # We replace the partial reward of the last move with the final reward of the game
        final_reward = self.env.compute_final_reward(self.trainer_side)
        rollout.update_last_reward(final_reward)

        if self.env.get_winner() == self.trainer_side:
            rollout.add_win()
        return rollout


class A3C(object):
    def __init__(self, env: MancalaEnv, task: int):
        self.env = env
        self.task = task

        # Performance statistics
        self.episodes_reward = []
        self.episodes_length = []
        self.episodes_mean_value = []
        self.wins = 0
        self.games = 0

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                # The input board is a tensor 2 x 8 x 1. The last dimension is added so that
                # convolutional layers can be applied to the input
                self.network = ACNetwork(state_shape=[2, 8, 1], num_act=7)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = self.network
                pi.global_step = self.global_step

            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_one_hot = tf.one_hot(self.action, 7, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)

            log_prob = tf.nn.log_softmax(pi.logits)
            prob = tf.nn.softmax(pi.logits)

            act_log_prob = tf.reduce_sum(log_prob * self.action_one_hot, [1])

            # Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(pi.value, [-1])))
            self.entropy = -tf.reduce_sum(prob * log_prob)
            self.policy_loss = -tf.reduce_sum(act_log_prob * self.advantage)
            # self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in local_vars])
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01  # + 0.002 * self.reg_loss

            # Get gradients from local network using local losses and clip them to avoid exploding gradients
            self.gradients = tf.gradients(self.loss, pi.vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 100.0)

            # Define operation for downloading the weights from the parameter server (ps)
            # on the local model of the worker
            self.down_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.vars, self.network.vars)])

            # Define the training operation which applies the gradients on the parameter server network (up sync)
            optimiser = tf.train.RMSPropOptimizer(learning_rate=0.0007)
            grads_and_global_vars = list(zip(grads, self.network.vars))
            inc_step = self.global_step.assign_add(tf.shape(self.action)[0])
            self.train_op = tf.group(*[optimiser.apply_gradients(grads_and_global_vars), inc_step])

            # Define an environment runner of this network
            self.env_runner = EnvironmentRunner(MancalaEnv(), pi)

            episode_size = tf.to_float(tf.shape(pi.value)[0])
            # Define summaries for tensorboard
            tf.summary.scalar("Model/PolicyLoss", self.policy_loss / episode_size)
            tf.summary.scalar("Model/ValueLoss", self.value_loss / episode_size)
            tf.summary.scalar("Model/Entropy", self.entropy / episode_size)
            tf.summary.scalar("Model/GradientsGlobalNorm", self.grad_norms)
            tf.summary.scalar("Model/VarGlobalNorm", tf.global_norm(pi.vars))
            self.summary_op = tf.summary.merge_all()

            self.summary_writer = None
            self.local_steps = 0

    def train(self, sess: tf.Session, rollout: Rollout, sum_period=100):
        # Record the statistics of this new rollout
        self.episodes_reward.append(np.sum(rollout.rewards))
        self.episodes_length.append(len(rollout.states))
        self.episodes_mean_value.append(np.mean(rollout.values))
        self.wins += rollout.win
        self.games += 1

        batch = generate_training_batch(rollout, gamma=0.99)

        feed_dict = {
            self.local_network.state: batch.states,
            self.action: batch.actions,
            self.advantage: batch.advantages,
            self.target_v: batch.discounted_rewards,
            self.local_network.mask: batch.masks,
        }

        should_compute_summary = self.task == 0 and self.local_steps % sum_period == 0
        if should_compute_summary:
            fetches = [self.train_op, self.global_step, self.summary_op]
        else:
            fetches = [self.train_op, self.global_step]
        fetched = sess.run(fetches, feed_dict)

        if should_compute_summary:
            # Keep only the last sum_period entries
            self.episodes_reward = self.episodes_reward[-sum_period:]
            self.episodes_length = self.episodes_length[-sum_period:]
            self.episodes_mean_value = self.episodes_mean_value[-sum_period:]

            # Add stats to tensorboard
            summary = tf.Summary()
            mean_reward = np.mean(self.episodes_reward[-sum_period:])
            mean_length = np.mean(self.episodes_length[-sum_period:])
            mean_value = np.mean(self.episodes_mean_value[-sum_period:])
            summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
            summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            summary.value.add(tag='Perf/WinRate', simple_value=float(self.wins / self.games))

            self.summary_writer.add_summary(summary, fetched[1])
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[2]), fetched[1])
            self.summary_writer.flush()

            # Restart the win rate statistics
            self.wins = self.games = 0
            self.summary_writer = None
        self.local_steps += 1

    def play(self, sess: tf.Session, opp_agent: Agent, summary_writer: tf.summary.FileWriter):
        self.summary_writer = summary_writer

        sess.run(self.down_sync)
        rollout = self.env_runner.run(sess, opp_agent)
        self.train(sess, rollout)

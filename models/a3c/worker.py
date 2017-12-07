import numpy as np
import tensorflow as tf
from models.a3c.ac_network import ActorCriticNetwork
from models.a3c.helpers import update_target_graph, discount, process_frame
from magent.mancala import MancalaEnv
from magent.side import Side
from magent.move import Move
import random


class Worker(object):
    def __init__(self, game: MancalaEnv, name, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("logs/a3c/train_" + str(self.number))
        self.a_size = a_size

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ActorCriticNetwork(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 3]
        masks = rollout[:, 4]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus[:-1], gamma)
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.valid_action_mask: np.vstack(masks),
                     self.local_AC.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def random_policy(self):
        action = np.random.choice(self.env.get_legal_moves())
        _ = self.env.perform_move(action)

    # def sample_action(self, logits, mask) -> (int, np.array):
    #     # If only one move is possible, then choose it
    #     if np.sum(mask) == 1.0:
    #         return np.argmax(mask), mask
    #
    #     logits = np.asarray(logits[0])
    #     exp_logits = np.exp(logits)
    #
    #     # If all of the legal moves have probability 0, assign to them a uniform distribution
    #     if np.sum(exp_logits * mask) < 1e-25:
    #         exp_logits = mask
    #     else:
    #         exp_logits *= mask
    #     dist = exp_logits / np.sum(exp_logits)
    #     return np.random.choice(range(7), p=dist), dist, exp_logits

    def pg_train_policy(self):
        # If there is only one possible move, just make the move. Evaluating network on this could destabilise weights.
        if len(self.env.get_legal_moves()) == 1:
            self.env.perform_move(self.env.get_legal_moves()[0])
            return

        # If the agent is playing as NORTH, it's input would be a flipped board
        flip_board = self.env.side_to_move == Side.NORTH
        state = self.env.board.get_board_image(flipped=flip_board)

        a_dist, v, logits = self.sess.run(
            [self.local_AC.policy, self.local_AC.value, self.local_AC.logits],
            feed_dict={self.local_AC.inputs: [state],
                       self.local_AC.valid_action_mask: [self.env.get_action_mask_with_no_pie()],
                       }
        )
        # Sample an action from the distribution
        action = np.random.choice(range(self.a_size), p=a_dist[0])

        # Due to numerical reasons, illegal actions might be sampled.
        # This is here for debugging reasons.
        if not self.env.is_legal(Move(self.agent_side, action + 1)):
            print(state)
            print(self.env.get_actions_mask())
            print(a_dist)
            print(v)
            print(logits)

        reward = 0
        try:
            seeds_in_store_before = self.env.board.get_seeds_in_store(self.agent_side)
            self.env.perform_move(Move(self.agent_side, action + 1))
            seeds_in_store_after = self.env.board.get_seeds_in_store(self.agent_side)
            reward = (seeds_in_store_after - seeds_in_store_before) / 100.0
        except ValueError:
            # Invalid move causes a lost game. This should not happen (usually)
            for hole in range(1, self.env.board.holes):
                self.env.board.set_seeds(self.agent_side, hole, 0)
            self.env.board.set_seeds_in_store(self.agent_side, 0)

        self.episode_buffer.append([[state], action, reward, v[0, 0], [self.env.get_actions_mask()]])
        self.episode_values.append(v[0, 0])

        self.episode_reward += reward
        self.total_steps += 1
        self.episode_step_count += 1

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        self.total_steps = 0
        self.played_games = 0
        self.won_games = 0
        self.sess = sess

        print("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                self.episode_buffer = []
                self.episode_values = []
                self.episode_reward = 0
                self.episode_step_count = 0

                self.env.reset()
                self.agent_side = Side.SOUTH  # if random.randint(0, 1) == 0 else Side.NORTH
                if self.agent_side == Side.SOUTH:
                    self.policy_rollout(self.pg_train_policy, self.random_policy)
                else:
                    self.policy_rollout(self.random_policy, self.pg_train_policy)

                # Add the final reward to the agent's list of rewards
                # If the agent didn't make the last move then the final reward must be added here.
                self.episode_reward += self.env.compute_reward(self.agent_side)  # / self.episode_step_count
                self.episode_buffer[-1][2] = self.env.compute_reward(self.agent_side)  # / self.episode_step_count

                self.played_games += 1
                if self.env.get_winner() == self.agent_side:
                    self.won_games += 1

                self.episode_rewards.append(self.episode_reward)
                self.episode_lengths.append(self.episode_step_count)
                self.episode_mean_values.append(np.mean(self.episode_values))

                # Update the network using the episode buffer at the end of the episode.
                v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 10 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    mean_value = np.mean(self.episode_mean_values[-10:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

                    if episode_count % 250 == 0:
                        summary.value.add(tag='Perf/WinRate', simple_value=float(self.won_games / self.played_games))
                        self.played_games = 0
                        self.won_games = 0

                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

    def policy_rollout(self, south_policy, north_policy):
        # Reset the environment to make sure everything starts in a clean state.
        self.env.reset()
        self.turns = 0
        while not self.env.is_game_over():
            self.turns += 1
            if self.env.side_to_move == Side.SOUTH:
                south_policy()
            else:
                north_policy()

            self.turns += 1

    def random_policy(self):
        action = np.random.choice(self.env.get_legal_moves())
        _ = self.env.perform_move(action)

import time

from magent.mancala import MancalaEnv
from models.reinforce_agent import PolicyGradientAgent
from magent.side import Side
from magent.move import Move
import numpy as np
from collections import deque
import tensorflow as tf
import random


class PolicyGradientTrainer(object):

    def __init__(self, agent: PolicyGradientAgent, env: MancalaEnv, period=200, backup_period=2000, games=1000000):
        self.agent = agent
        self.env = env
        self.period = period
        self.backup_period = backup_period
        self.games_to_play = games

        self.turns_history = deque([], 10)
        self.games = 0
        self.wins = 0

        # Tensorboard statistics
        self.win_rate_sum = tf.Summary()
        self.avg_turns_sum = tf.Summary()
        self.avg_reward_sum = tf.Summary()

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

        self.turns_history.append(self.turns)

    def random_policy(self):
        action = np.random.choice(self.env.get_legal_moves())
        _ = self.env.perform_move(action)

    def pg_train_policy(self):
        flip_board = self.env.side_to_move == Side.NORTH
        state = self.env.board.get_board_image(flipped=flip_board)
        valid_actions_mask = self.env.get_actions_mask()

        action = self.agent.sample_action(state, valid_actions_mask)

        seeds_in_store_before = self.env.board.get_seeds_in_store(self.agent_side)
        self.env.perform_move(Move(self.agent_side, action))

        seeds_in_store_after = self.env.board.get_seeds_in_store(self.agent_side)
        reward = (seeds_in_store_after - seeds_in_store_before) / 10.0
        self.agent.store_rollout(state, action, reward, valid_actions_mask)

    def train(self):
        start_time = time.time()
        for game_no in range(self.games_to_play):
            self.agent_side = Side.SOUTH if random.randint(0, 1) == 0 else Side.NORTH
            if self.agent_side == Side.SOUTH:
                self.policy_rollout(self.pg_train_policy, self.random_policy)
            else:
                self.policy_rollout(self.random_policy, self.pg_train_policy)

            # Add the final reward to the agent's list of rewards
            # If the agent didn't make the last move then the final reward must be added here.
            self.agent.rewards[-1] = self.env.compute_reward(self.agent_side) / self.turns
            self.agent.run_train_step(game_no)

            self.games += 1
            if self.env.get_winner() == self.agent_side:
                self.wins += 1

            if game_no % self.period == 0:
                running_time = (time.time() - start_time) / 60.0
                print('Games played: [%2d] Elapsed minutes: %4.4f' % (game_no, running_time))
                print('Agent winning rate {}'.format(self.wins/self.games))

                self.win_rate_sum.value.add(tag='win_rate', simple_value=self.wins/self.games)
                self.agent.writer.add_summary(self.win_rate_sum, game_no)

                self.avg_turns_sum.value.add(tag='avg_turns_sum', simple_value=np.mean(self.turns_history))
                self.agent.writer.add_summary(self.avg_turns_sum, game_no)

                self.avg_reward_sum.value.add(tag='avg_reward_sum', simple_value=np.mean(self.agent.get_average_reward()))
                self.agent.writer.add_summary(self.avg_reward_sum, game_no)

                # Restart statistics every few games
                self.wins = 0
                self.games = 0

            if game_no % self.backup_period == 0:
                self.agent.save_model_params(self.agent.name)






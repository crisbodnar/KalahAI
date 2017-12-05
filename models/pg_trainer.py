import time

from magent.mancala import MancalaEnv
from models.reinforce_agent import PolicyGradientAgent
from magent.side import Side
from magent.move import Move
import numpy as np
from collections import deque
import tensorflow as tf


class PolicyGradientTrainer(object):

    def __init__(self, agent: PolicyGradientAgent, opponent: PolicyGradientAgent, env: MancalaEnv):
        self.agent = agent
        self.opponent = opponent
        self.env = env

        self.turns_history = deque([], 10)
        self.games = 0
        self.south_wins = 0

    def policy_rollout(self):
        # Reset the environment to make sure everything starts in a clean state.
        self.env.reset()
        self.turns = 0
        while not self.env.is_game_over():
            self.turns += 1
            if self.env.side_to_move == Side.SOUTH:
                state = self.env.board.get_board_image()
                valid_actions_mask = self.env.get_actions_mask()
                action = self.agent.sample_action(state, valid_actions_mask)
                reward = self.env.perform_move(Move(Side.SOUTH, action))
                self.agent.store_rollout(state, action, reward, valid_actions_mask)
            else:
                action = np.random.choice(self.env.get_legal_moves())
                _ = self.env.perform_move(action)

        if self.env.get_winner() == Side.SOUTH:
            self.agent.rewards[-1] = 500
        elif self.env.get_winner() == Side.NORTH:
            self.agent.rewards[-1] = -500

        self.turns_history.append(self.turns)
        self.games += 1
        self.south_wins += 1 if self.env.get_winner() == Side.SOUTH else 0

    def play_against_random(self) -> float:
        south_wins = 0
        for g in range(100):
            self.env.reset()
            while not self.env.is_game_over():
                self.turns += 1
                if self.env.side_to_move == Side.SOUTH:
                    state = self.env.board.get_board_image()
                    valid_actions_mask = self.env.get_actions_mask()
                    action = self.agent.get_best_action(state, valid_actions_mask)
                    _ = self.env.perform_move(Move(Side.SOUTH, action))
                else:
                    action = np.random.choice(self.env.get_legal_moves())
                    _ = self.env.perform_move(action)
            south_wins += 1 if self.env.get_winner() == Side.SOUTH else 0
        return south_wins / 100.0

    def train(self, games=100000):
        start_time = time\
            .time()
        for game_no in range(games):
            self.policy_rollout()
            south_loss = self.agent.run_train_step(game_no)

            if game_no % 200 == 0:
                print('South winning rate {}'.format(self.south_wins/self.games))

                avg_turns_sum = tf.Summary()
                avg_turns_sum.value.add(tag='avg_turns_sum', simple_value=np.mean(self.turns_history))
                self.agent.writer.add_summary(avg_turns_sum, game_no)

                avg_reward_sum = tf.Summary()
                avg_reward_sum.value.add(tag='avg_reward_sum', simple_value=np.mean(self.agent.get_average_reward()))
                self.agent.writer.add_summary(avg_reward_sum, game_no)

                # Restart statistics every few games
                self.south_wins = 0
                self.games = 0

            if game_no % 2000 == 0:
                self.agent.transfer_params(self.opponent)
                self.agent.save_model_params('south')

                rand_game_sum = tf.Summary()
                rand_game_sum.value.add(tag='rand_game_win_rate', simple_value=self.play_against_random())
                self.agent.writer.add_summary(rand_game_sum, game_no)






from magent.mancala import MancalaEnv
from models.pg_reinforce import PolicyGradientAgent
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
                action = self.agent.sample_action(np.array(state), valid_actions_mask)
                reward = self.env.perform_move(Move(Side.SOUTH, action))
                self.agent.store_rollout(state, action, reward, valid_actions_mask)
            else:
                state = self.env.board.get_board_image(flipped=True)
                valid_actions_mask = self.env.get_actions_mask()
                action = self.opponent.sample_action(state, valid_actions_mask)
                _ = self.env.perform_move(Move(Side.NORTH, action))

        self.turns_history.append(self.turns)
        self.games += 1
        self.south_wins += 1 if self.env.get_winner() == Side.SOUTH else 0

    def train(self, games=100000):
        for t in range(games):
            self.policy_rollout()
            south_loss = self.agent.run_train_step()

            if t % 100 == 0:
                print('Agent loss: {} | Average reward: {}'.format(south_loss, self.agent.get_average_reward()))
                print('Avg number of turns: {}'.format(np.mean(self.turns_history)))
                print('South winning rate {}'.format(self.south_wins/self.games))

                # Restart statistics every few games
                self.south_wins = 0
                self.games = 0

            if t % 1000 == 0:
                self.agent.transfer_params(self.opponent)
                self.agent.save_model_params('agent')




from magent.mancala import MancalaEnv
from models.pg_reinforce import PolicyGradientAgent
from magent.side import Side
from magent.move import Move
import numpy as np
from collections import deque

class PolicyGradientTrainer(object):

    def __init__(self, south: PolicyGradientAgent, north: PolicyGradientAgent, env: MancalaEnv):
        self.south = south
        self.north = north
        self.env = env

        self.turns_history = deque([], 10)

    def policy_rollout(self):
        # Reset the environment to make sure everything starts in a clean state.
        self.env.reset()
        self.turns = 0
        while not self.env.is_game_over():
            self.turns += 1
            state = self.env.board.get_board_image()
            if self.env.side_to_move == Side.SOUTH:
                valid_actions_mask = self.env.get_valid_actions_mask()
                action = self.south.sample_action(np.array(state), valid_actions_mask)
                reward = self.env.perform_move(Move(Side.SOUTH, action))
                self.south.store_rollout(state, action, reward, valid_actions_mask)
            else:
                valid_actions_mask = self.env.get_valid_actions_mask()
                action = self.north.sample_action(state, valid_actions_mask)
                reward = self.env.perform_move(Move(Side.NORTH, action))
                self.north.store_rollout(state, action, reward, valid_actions_mask)

        self.turns_history.append(self.turns)

    def train(self, games=10000):
        for t in range(games):
            self.policy_rollout()
            south_loss = self.south.run_train_step()
            north_loss = self.north.run_train_step()

            if t % 100 == 0:
                print('South loss: {} | Average reward: {}'.format(south_loss, self.south.get_average_reward()))
                print('North loss: {} | Average reward: {}'.format(north_loss, self.north.get_average_reward()))
                print('Avg number of turns: {}'.format(np.mean(self.turns_history)))




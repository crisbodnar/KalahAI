from magent.mancala import MancalaEnv
from models.pg_reinforce import PolicyGradientAgent
from magent.side import Side
from magent.move import Move
import numpy as np


class PolicyGradientTrainer(object):
    def __init__(self, south: PolicyGradientAgent, north: PolicyGradientAgent, env: MancalaEnv):
        self.south = south
        self.north = north
        self.env = env

    def policy_rollout(self):
        # Reset the environment to make sure everything starts in a clean state.
        self.env.reset()

        while not self.env.is_game_over():
            state = self.env.board.board
            if self.env.side_to_move is Side.SOUTH:
                valid_actions_mask = self.env.get_valid_actions_mask(self.env.get_legal_moves())
                action = self.south.sample_action(np.array(self.env.board.board), valid_actions_mask)
                reward = self.env.perform_move(Move(Side.SOUTH, action))
                self.south.store_rollout(state, action, reward, valid_actions_mask)
            else:
                valid_actions_mask = self.env.get_valid_actions_mask(self.env.get_legal_moves())
                action = self.north.sample_action(np.array(self.env.board.board), valid_actions_mask)
                reward = self.env.perform_move(Move(Side.NORTH, action))
                self.north.store_rollout(state, action, reward, valid_actions_mask)

    def train(self, games=10000):
        for t in range(games):
            self.policy_rollout()
            self.south.run_train_step()
            self.north.run_train_step()




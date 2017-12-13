from magent.mancala import MancalaEnv
from magent.side import Side
import numpy as np
import random
import math


class TDAgent(object):
    def __init__(self, model, name: str = 'TDAgent'):
        self.model = model
        self.name = name

    def get_action(self, env: MancalaEnv, explore=False, exploration_const=1):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """

        side = env.side_to_move
        actions = env.get_legal_moves()

        v_best = 0
        a_best = np.random.choice(actions)

        if explore:
            exploration_const = int(1 + math.sqrt(exploration_const) / 20)
            if random.randint(1, exploration_const) == 1:
                return a_best

        for a in actions:
            clone = MancalaEnv.clone(env)
            clone.perform_move(a)

            board = clone.board.get_board_image_with_heuristics(clone.side_to_move)
            v = self.model.get_output(board)

            v = 1. - v if side == Side.NORTH else v
            if v >= v_best:
                v_best = v
                a_best = a
        return a_best


class RandomAgent(object):
    def __init__(self, name='RandomAgent'):
        self.name = name

    def get_action(self, env: MancalaEnv):
        return np.random.choice(env.get_legal_moves())

import numpy as np

from magent.mancala import MancalaEnv
from magent.move import Move
from magent.side import Side
from magent.treesearch.treesearch import TreeSearch


class AlphaBeta(TreeSearch):
    def __init__(self, depth):
        self.depth = depth

    def search(self, game: MancalaEnv) -> Move:
        values = [(a, self.alpha_beta_search(game=state, depth=self.depth)) for a, state in game.next_states()]
        np.random.shuffle(values)

        if game.side_to_move == Side.SOUTH:
            action, _ = max(values, key=lambda x: x[1])
        else:
            action, _ = min(values, key=lambda x: x[1])
        return action

    @staticmethod
    def _alpha_beta_search(game: MancalaEnv, alpha=-np.inf, beta=np.inf, depth=5):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""
        if depth == 0 or game.is_game_over():
            return game.get_player_utility()

        if game.side_to_move == Side.SOUTH:
            v = -np.inf
            for (_, new_s) in game.next_states():
                v = max(v, AlphaBeta._alpha_beta_search(new_s, alpha, beta, depth - 1))
                alpha = max(alpha, v)
                # if beta <= alpha:
                #     break
        else:
            v = np.inf
            for (_, new_s) in game.next_states():
                v = min(v, AlphaBeta._alpha_beta_search(new_s, alpha, beta, depth - 1))
                beta = min(beta, v)
                # if beta <= alpha:
                #     break
        return v

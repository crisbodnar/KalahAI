import datetime
import logging

from magent.mancala import MancalaEnv
from magent.side import Side
import numpy as np

logfile_name = 'here.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./logs/' + logfile_name,
                    filemode='w')


def alpha_beta_search(game: MancalaEnv, depth=5):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    
    infinity = 100000000

    def max_value(_state: MancalaEnv, alpha: int, beta: int, _depth: int) -> int:
        assert _state.side_to_move == Side.SOUTH
        if cutoff_test(_state, _depth):
            return _state.get_player_utility()
        v = alpha
        for (_, _s) in _state.next_states():
            if _s.side_to_move == Side.SOUTH:
                v = max(v, max_value(_s, alpha, beta, _depth + 1))
            else:
                v = max(v, min_value(_s, alpha, beta, _depth + 1))
            alpha = max(alpha, v)
            if beta <= alpha:
                return v
        return v

    def min_value(_state, alpha, beta, _depth):
        assert _state.side_to_move == Side.NORTH
        if cutoff_test(_state, _depth):
            return _state.get_player_utility()

        v = beta
        for (_, _s) in _state.next_states():
            if _s.side_to_move == Side.NORTH:
                v = min(v, min_value(_s, alpha, beta, _depth + 1))
            else:
                v = min(v, max_value(_s, alpha, beta, _depth + 1))
            beta = min(beta, v)
            if beta <= alpha:
                return v
        return v

    value_next_states = []
    for a, s in game.next_states():
        if s.side_to_move == Side.SOUTH:
            value_next_states.append((a, max_value(s, -infinity, infinity, 0)))
        else:
            value_next_states.append((a, min_value(s, -infinity, infinity, 0)))

    logging.info("Values: %s" % value_next_states)
    if game.side_to_move == Side.SOUTH:
        maximum = -infinity
        act = None
        for action, val in value_next_states:
            if val > maximum:
                maximum = val
                act = action
        logging.info("Maximum: %d" % maximum)
        return act
    else:
        minimum = infinity
        act = None
        for action, val in value_next_states:
            if val < minimum:
                minimum = val
                act = action
        logging.info("Minimum: %d" % minimum)
        return act

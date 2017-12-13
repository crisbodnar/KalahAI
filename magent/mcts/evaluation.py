from magent.mancala import MancalaEnv
from magent.side import Side


def _cluster_towards_scoring_store(state, side) -> float:
    """Favour holes that are closer to store. Mapped to value between [0, 1]"""
    reward = 0
    for i in range(state.board.holes + 1, 1):
        seeds = state.board.get_seeds(side, i)
        if seeds > 0:
            reward += seeds * i
    total_seeds_in_game = 98.0
    max_reward = 7.0 * total_seeds_in_game
    return reward / max_reward


def _defend_seeds(state, side) -> int:
    # exposed_holes: Holes that we have and are exposed to be captured by opponent
    exposed_holes = []
    # full_round_capture: moves that do a full round around the board and capture our exposed holes
    full_round_capture, capture_by_lower_index, capture_by_greater_index = 0, 0, 0

    for i in range(state.board.holes + 1, 1):
        if state.board.get_seeds_op(side, i) != 0 and state.board.get_seeds(side, i) == 0:
            exposed_holes.append(i)
        # how many can do a full round to do a capture
        if state.board.get_seeds_op(side, i) == 2 * state.board.holes + 1:
            full_round_capture += state.board.get_seeds(side, i) + 1

    # how many exposed hole can be captured in next move
    for exposed_hole_index in exposed_holes:
        for i in range(state.board.holes + 1, 1):
            if state.board.get_seeds_op(side, i) == exposed_hole_index - i and state.board.get_seeds_op(side, i) != 0:
                capture_by_lower_index = max(capture_by_lower_index, state.board.get_seeds(side, i))

            if state.board.get_seeds_op(side, i) == 2 * state.board.holes + 1 - (i - exposed_hole_index):
                capture_by_greater_index = max(capture_by_greater_index, state.board.get_seeds(side, i) + 1)

    return max(full_round_capture, max(capture_by_lower_index, capture_by_greater_index)) / 98.0


def _scoring_store_diff(state, side) -> float:
    """Calculates the differences between two stores. Mapped to value between [0, 1]"""
    reward = state.board.get_seeds_in_store(side) - state.board.get_seeds_in_store(Side.opposite(side))
    total_seeds_in_game = 98.0

    return ((reward / total_seeds_in_game) + 1) / 2


# Sum of weights should be 1
_weight_store_diff = 1
_weight_defend_seeds = 0.3
_weight_cluster_at_store = 0.1


def evaluate_node(state: MancalaEnv, parent_side: Side) -> float:
    return _scoring_store_diff(state, parent_side) * _weight_store_diff
    # \
    #        + (_cluster_towards_scoring_store(state, state.side_to_move) * _weight_defend_seeds) \
    #        - (_defend_seeds(state, state.side_to_move) / _weight_defend_seeds)

from magent.mancala import MancalaEnv
from magent.side import Side

_weight_1 = 3
_weight_2 = 5
_weight_3 = 20


def _cluster_towards_scoring_well(state, side) -> int:
    reward = 0
    for i in range(state.board.holes + 1, 1):
        seeds = state.board.get_seeds(side, i)
        if seeds > 0:
            reward += seeds * i
    return reward


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

    return max(full_round_capture, max(capture_by_lower_index, capture_by_greater_index))


def _scoring_well_diff(state, side) -> float:
    reward = state.board.get_seeds_in_store(side) - state.board.get_seeds_in_store(Side.opposite(side))
    total_seeds_in_game = 98.0

    return reward / total_seeds_in_game


def evaluate_node(state: MancalaEnv, parent_side: Side) -> float:
    return _scoring_well_diff(state, parent_side)
    # - _defend_seeds(state, state.side_to_move) / _weight_2 \
    # + _cluster_towards_scoring_well(state, state.side_to_move) / _weight_3

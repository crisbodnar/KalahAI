from magent.mancala import MancalaEnv
from magent.side import Side


def _cluster_towards_scoring_store(state: MancalaEnv, parent_side: Side) -> float:
    """Favour holes that are closer to store."""
    reward = 0.0
    for i in range(state.board.holes + 1, 1):
        seeds = state.board.get_seeds(parent_side, i)
        if seeds > 0:
            reward += seeds * i
    return reward


def _defend_seeds(state: MancalaEnv, parent_side: Side) -> int:
    # exposed_holes: Holes that we have and are exposed to be captured by opponent
    exposed_holes = []
    # full_round_capture: moves that do a full round around the board and capture our exposed holes
    full_round_capture, capture_by_lower_index, capture_by_greater_index = 0, 0, 0

    for i in range(1, state.board.holes + 1, 1):
        if state.board.get_seeds_op(parent_side, i) == 0 and state.board.get_seeds(parent_side, i) != 0:
            exposed_holes.append(i)
        # how many can do a full round to do a capture
        if state.board.get_seeds_op(parent_side, i) == 2 * state.board.holes + 1:
            full_round_capture += state.board.get_seeds(parent_side, i) + 1

    # how many exposed hole can be captured in next move
    for exposed_hole_index in exposed_holes:
        for i in range(1, state.board.holes + 1, 1):
            if state.board.get_seeds_op(parent_side, i) == exposed_hole_index - i \
                    and state.board.get_seeds_op(parent_side, i) != 0:
                capture_by_lower_index = max(capture_by_lower_index, state.board.get_seeds(parent_side, i))

            if state.board.get_seeds_op(parent_side, i) == 2 * state.board.holes + 1 - (i - exposed_hole_index):
                capture_by_greater_index = max(capture_by_greater_index, state.board.get_seeds(parent_side, i) + 1)

    return max(full_round_capture, max(capture_by_lower_index, capture_by_greater_index + 1))


def _stones_in_holes_diff(state: MancalaEnv, parent_side: Side) -> int:
    our_seeds = 0
    their_seeds = 0
    for i in range(1, state.board.holes + 1, 1):
        our_seeds += state.board.get_seeds(parent_side, i)
        their_seeds += state.board.get_seeds_op(parent_side, i)
    reward = our_seeds - their_seeds

    return reward


def _scoring_store_diff(state: MancalaEnv, parent_side: Side) -> int:
    """Calculates the differences between two stores."""
    our_seeds = state.board.get_seeds_in_store(parent_side)
    their_seeds = state.board.get_seeds_in_store(Side.opposite(parent_side))

    reward = our_seeds - their_seeds

    return reward


_weight_store_diff = 3.0
_weight_cluster_at_store = 5.0
_weight_defend_seeds = 20.0


def evaluate_node(state: MancalaEnv, parent_side: Side) -> float:
    return _scoring_store_diff(state, parent_side) * _weight_store_diff \
           - (_defend_seeds(state, parent_side) / _weight_defend_seeds) \
           + (_cluster_towards_scoring_store(state, parent_side) / _weight_cluster_at_store)


_weight_kalah_location = 8.0
_weight_extra_turn = 4.0
_weight_capture_weight = 0.5


def _can_put_last_seed_here(state: MancalaEnv, parent_side: Side, index_of_last_hole: int) -> bool:
    for i in range(1, index_of_last_hole, 1):
        if state.board.get_seeds(parent_side, i) == index_of_last_hole - i:
            return True
    return False


def _compute_holes_captures(state: MancalaEnv, parent_side: Side) -> int:
    results = 0
    for i in range(1, state.board.holes + 1, 1):
        if state.board.get_seeds(parent_side, i) == 0 and _can_put_last_seed_here(state, parent_side, i):
            results += state.board.get_seeds_op(parent_side, i) + 1
    return results


def _can_get_extra_turn(state: MancalaEnv, parent_side: Side) -> bool:
    for i in range(1, state.board.holes + 1, 1):
        if state.board.get_seeds(parent_side, i) == state.board.holes + 1 - i:
            return True
    return False


def get_score(state: MancalaEnv, parent_side: Side) -> float:
    reward = _compute_holes_captures(state, parent_side) * _weight_capture_weight
    if _can_get_extra_turn(state, parent_side):
        reward = reward * _weight_extra_turn
    if parent_side != state.our_side:
        reward = reward * -1  # punish bad moves that hurt us
    reward += _scoring_store_diff(state, parent_side)
    return reward

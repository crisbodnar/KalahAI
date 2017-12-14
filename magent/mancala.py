from copy import deepcopy
from typing import List

import numpy as np

from magent.board import Board
from magent.move import Move
from magent.side import Side


class MancalaEnv(object):
    def __init__(self):
        self.reset()

    @property
    def board(self):
        return self._board

    @board.setter
    def board(self, board: Board):
        self._board = board

    @property
    def side_to_move(self):
        return self._side_to_move

    @side_to_move.setter
    def side_to_move(self, side: Side):
        self._side_to_move = side

    @property
    def north_moved(self):
        return self._north_moved

    @north_moved.setter
    def north_moved(self, moved: bool):
        self._north_moved = moved

    @property
    def our_side(self):
        return self._my_side

    @our_side.setter
    def our_side(self, side: Side):
        self._my_side = side

    def reset(self):
        self.board = Board(7, 7)
        self.side_to_move = Side.SOUTH
        self.north_moved = False
        self.our_side = Side.SOUTH

    @staticmethod
    def clone(other_state):
        board = Board.clone(other_state.board)
        side_to_move = deepcopy(other_state.side_to_move)
        north_moved = deepcopy(other_state.north_moved)

        clone_game = MancalaEnv()
        clone_game.board = board
        clone_game.side_to_move = side_to_move
        clone_game.north_moved = north_moved
        return clone_game

    def get_legal_moves(self) -> List[Move]:
        return MancalaEnv.get_state_legal_actions(self.board, self.side_to_move, self.north_moved)

    def is_legal(self, move: Move) -> bool:
        return MancalaEnv.is_legal_action(self.board, move, self.north_moved)

    def perform_move(self, move: Move) -> int:
        """Performs a move and returns the reward for this move."""
        seeds_in_store_before = self.board.get_seeds_in_store(move.side)
        if move.index == 0:  # pie move
            self.our_side = Side.opposite(self.our_side)
        self.side_to_move = MancalaEnv.make_move(self.board, move, self.north_moved)
        if move.side == Side.NORTH:
            self.north_moved = True
        seeds_in_store_after = self.board.get_seeds_in_store(move.side)

        # Return a partial reward proportional to the number of captured seeds.
        return (seeds_in_store_after - seeds_in_store_before) / 100.0

    def compute_final_reward(self, side: Side):
        """Returns a reward for the specified side for moving to the current state."""
        reward = self.board.get_seeds_in_store(side) - self.board.get_seeds_in_store(Side.opposite(side))
        return reward

    def compute_end_game_reward(self, side: Side):
        """Returns a reward for the specified side for moving to the end game state."""
        if not self.is_game_over():
            raise ValueError("compute_end_game_reward should only be called at end of the game")

        reward = self.compute_final_reward(side)
        if reward > 0:
            return 1  # win
        elif reward < 0:
            return 0  # lose
        return 0.5  # tie

    def is_game_over(self) -> bool:
        return MancalaEnv.game_over(self.board)

    def get_actions_mask(self) -> [float]:
        """Returns an np array of 1s and 0s where 1 at index i means that the action with that action is valid. """
        mask = [0 for _ in range(self.board.holes + 1)]
        moves = self.get_legal_moves()
        for action in moves:
            mask[action.index] = 1
        return np.array(mask)

    def get_action_mask_with_no_pie(self) -> [float]:
        """
        Returns an np array of 1s and 0s where 1 at index i means that the action with that action is valid.
        The pie move is not considered.
        """
        mask = [0 for _ in range(self.board.holes)]
        moves = [move.index for move in self.get_legal_moves()]
        if 0 in moves:
            moves.remove(0)
        for action in moves:
            mask[action - 1] = 1
        return np.array(mask)

    def get_winner(self) -> Side or None:
        """
        :return: The winning Side of the game or none if there is a tie.
        """
        if not self.is_game_over():
            raise ValueError('This method should be called only when the game is over')
        finished_side = Side.NORTH if MancalaEnv.holes_empty(self.board, Side.NORTH) else Side.SOUTH

        not_finished_side = Side.opposite(finished_side)
        not_finished_side_seeds = self.board.get_seeds_in_store(not_finished_side)
        for hole in range(1, self.board.holes + 1):
            not_finished_side_seeds += self.board.get_seeds(not_finished_side, hole)
        finished_side_seeds = self.board.get_seeds_in_store(finished_side)

        if finished_side_seeds > not_finished_side_seeds:
            return finished_side
        elif finished_side_seeds < not_finished_side_seeds:
            return not_finished_side
        return None

    # Generate a list of all legal moves given a board state and a side
    @staticmethod
    def get_state_legal_actions(board: Board, side: Side, north_moved: bool) -> List[Move]:
        # If this is the first move of NORTH, then NORTH can use the pie rule action
        legal_moves = [] if north_moved or side == side.SOUTH else [Move(side, 0)]
        for i in range(1, board.holes + 1):
            if board.board[side.get_index(side)][i] > 0:
                legal_moves.append(Move(side, i))
        return legal_moves

    @staticmethod
    def is_legal_action(board: Board, move: Move, north_moved: bool) -> bool:
        return move.index in [act.index for act in MancalaEnv.get_state_legal_actions(board, move.side, north_moved)]

    @staticmethod
    def holes_empty(board: Board, side: Side):
        for hole in range(1, board.holes + 1):
            if board.get_seeds(side, hole) > 0:
                return False
        return True

    @staticmethod
    def game_over(board: Board):
        """
        :param board: The board to be analysed
        :return: True if the game is over and the side which finished
        """
        if MancalaEnv.holes_empty(board, Side.SOUTH):
            return True
        if MancalaEnv.holes_empty(board, Side.NORTH):
            return True
        return False

    @staticmethod
    def switch_sides(board: Board):
        for hole in range(board.holes + 1):
            board.board[0][hole], board.board[1][hole] = board.board[1][hole], board.board[0][hole]

    @staticmethod
    def make_move(board: Board, move: Move, north_moved):
        if not MancalaEnv.is_legal_action(board, move, north_moved):
            raise ValueError('Move is illegal: Board: \n {} \n Move:\n {}/{} \n {}'.format(board, move.index, move.side,
                                                                                           north_moved))

        # This is a pie move
        if move.index == 0:
            MancalaEnv.switch_sides(board)
            return Side.opposite(move.side)

        seeds_to_sow = board.get_seeds(move.side, move.index)
        board.set_seeds(move.side, move.index, 0)

        holes = board.holes
        # Place seeds in all holes excepting the opponent's store
        receiving_holes = 2 * holes + 1
        # Rounds needed to sow all the seeds
        rounds = seeds_to_sow // receiving_holes
        # Seeds remaining after all the rounds
        remaining_seeds = seeds_to_sow % receiving_holes

        # Sow the seeds for the full rounds
        if rounds != 0:
            for hole in range(1, holes + 1):
                board.add_seeds(Side.NORTH, hole, rounds)
                board.add_seeds(Side.SOUTH, hole, rounds)
            board.add_seeds_to_store(move.side, rounds)

        # Sow the remaining seeds
        sow_side = move.side
        sow_hole = move.index
        for _ in range(remaining_seeds):
            sow_hole += 1
            if sow_hole == 1:
                sow_side = Side.opposite(sow_side)
            if sow_hole > holes:
                if sow_side == move.side:
                    sow_hole = 0
                    board.add_seeds_to_store(sow_side, 1)
                    continue
                else:
                    sow_side = Side.opposite(sow_side)
                    sow_hole = 1
            board.add_seeds(sow_side, sow_hole, 1)

        # Capture the opponent's seeds from the opposite hole if the last seed
        # is placed in an empty hole and there are seeds in the opposite hole
        if sow_side == move.side and sow_hole > 0 \
                and board.get_seeds(sow_side, sow_hole) == 1 \
                and board.get_seeds_op(sow_side, sow_hole) > 0:
            board.add_seeds_to_store(move.side, 1 + board.get_seeds_op(sow_side, sow_hole))
            board.set_seeds(move.side, sow_hole, 0)
            board.set_seeds_op(move.side, sow_hole, 0)

        # If the game is over, collect the seeds not in the store and put them there
        game_over = MancalaEnv.game_over(board)
        if game_over:
            finished_side = Side.NORTH if MancalaEnv.holes_empty(board, Side.NORTH) else Side.SOUTH
            seeds = 0
            collecting_side = Side.opposite(finished_side)
            for hole in range(1, board.holes + 1):
                seeds += board.get_seeds(collecting_side, hole)
                board.set_seeds(collecting_side, hole, 0)
            board.add_seeds_to_store(collecting_side, seeds)

        # Return the side which is next to move
        if sow_hole == 0 and (move.side == Side.NORTH or north_moved):
            return move.side  # Last seed was placed in the store, so side moves again
        return Side.opposite(move.side)

    def get_player_utility(self) -> int:
        # delta_defend = _defend_seeds(self, Side.SOUTH) - _defend_seeds(self, Side.NORTH)
        # more_than_half_in_store_south = 1000 if self.board.get_seeds_in_store(Side.SOUTH) / 98.0 > 0.5 else 0
        # more_than_half_in_store_north = 1000 if self.board.get_seeds_in_store(Side.NORTH) / 98.0 > 0.5 else 0

        store_score = compute_store_score(self)
        capture_score = compute_score_capture_by(self, Side.SOUTH) - compute_score_capture_by(self, Side.NORTH)
        double_move_score = compute_double_moves_score(self, Side.SOUTH) - compute_double_moves_score(self, Side.NORTH)
        delta_side_score = (compute_seeds_on_side(self, Side.SOUTH) - compute_seeds_on_side(self, Side.NORTH)) / 2

        # print(self)
        # print(store_score)
        # print(capture_score)
        # print(double_move_score)
        # print(delta_side_score)
        # print('==================================')

        return store_score + capture_score + double_move_score + delta_side_score

    def next_states(self):
        actions = self.get_legal_moves()

        next_states = []
        for action in actions:
            clone = MancalaEnv.clone(self)
            clone.perform_move(action)
            next_states.append((action, clone))
        return next_states

    def __hash__(self) -> int:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163]

        hashkey = 0
        hashkey += primes[0] * Side.get_index(self.side_to_move)
        hashkey += primes[1] * int(self.north_moved)
        for hole in range(self.board.holes + 1):
            hashkey += primes[2 + hole] * self.board.board[0][hole]
            hashkey += primes[10 + hole] * self.board.board[1][hole]
        return hashkey

    def __str__(self):
        return "%s" % self.board


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
            if state.board.get_seeds_op(side, i) == exposed_hole_index - i and state.board.get_seeds_op(side,
                                                                                                        i) != 0:
                capture_by_lower_index = max(capture_by_lower_index, state.board.get_seeds(side, i))

            if state.board.get_seeds_op(side, i) == 2 * state.board.holes + 1 - (i - exposed_hole_index):
                capture_by_greater_index = max(capture_by_greater_index, state.board.get_seeds(side, i) + 1)

    return max(full_round_capture, max(capture_by_lower_index, capture_by_greater_index))


def compute_store_score(game):
    if game.board.get_seeds_in_store(Side.SOUTH) == 0 and game.board.get_seeds_in_store(Side.NORTH) == 0:
        return 0
    if game.board.get_seeds_in_store(Side.SOUTH) == game.board.get_seeds_in_store(Side.NORTH) == 0:
        return 0
    max_store = max(game.board.get_seeds_in_store(Side.SOUTH), game.board.get_seeds_in_store(Side.NORTH))
    min_store = min(game.board.get_seeds_in_store(Side.SOUTH), game.board.get_seeds_in_store(Side.NORTH))
    score = 2 * max_store - min_store
    if max_store == game.board.get_seeds_in_store(Side.NORTH):
        score *= -1
    return score


def compute_score_capture_by(game, side: Side):
    board = game.board
    score = 0
    for hole in range(1, board.holes + 1):
        if board.get_seeds(side, hole) == 0 and board.is_seedable(side, hole):
            score += board.get_seeds_op(side, hole) / 2
    return score


def compute_double_moves_score(game, side: Side):
    board = game.board
    score = 0
    for hole in range(1, board.holes + 1):
        if board.holes + 1 - hole == board.get_seeds(side, hole):
            score += 1
    return score


def compute_seeds_on_side(game, side: Side):
    board = game.board
    seeds = 0
    for hole in range(1, board.holes + 1):
        seeds += board.get_seeds(side, hole)
    return seeds
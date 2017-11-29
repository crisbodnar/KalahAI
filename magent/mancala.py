from magent.board import Board
from magent.move import Move
from magent.side import Side
from copy import deepcopy
from typing import List
import numpy as np


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

    def reset(self):
        self.board = Board(7, 7)
        self.side_to_move = Side.SOUTH
        self.north_moved = False

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
        return MancalaEnv.get_state_legal_moves(self.board, self.side_to_move, self.north_moved)

    def is_legal(self, move: Move) -> bool:
        return MancalaEnv.is_legal_move(self.board, move, self.north_moved)

    def perform_move(self, move: Move) -> int:
        """Performs a move and returns the reward for this move."""
        self.side_to_move = MancalaEnv.make_move(self.board, move, self.north_moved)
        if move.side is Side.NORTH:
            self.north_moved = True

        return self.compute_reward(move.side)

    def compute_reward(self, side: Side):
        """Returns a reward for the specified side for moving to the current state."""
        if self.is_game_over():
            return 1000 if self.get_winner() is side else -1000
        reward = self.board.get_seeds_in_store(Side.NORTH) - self.board.get_seeds_in_store(Side.SOUTH)
        return reward if side is Side.NORTH else -reward

    def is_game_over(self) -> bool:
        return MancalaEnv.game_over(self.board)

    def get_valid_actions_mask(self) -> [float]:
        """Returns an np array of 1s and 0s where 1 at index i means that the action with that action is valid. """
        mask = [100000 for _ in range(self.board.holes + 1)]
        for action in self.get_legal_moves():
            mask[action.index] = 0
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

    # Generate a set of all legal moves given a board state and a side
    @staticmethod
    def get_state_legal_moves(board: Board, side: Side, north_moved: bool) -> List[Move]:
        # If this is the first move of NORTH, then NORTH can use the pie rule action
        legal_moves = [] if north_moved or side is side.SOUTH else [Move(side, 0)]
        for i in range(1, board.holes + 1):
            if board.board[side.get_index(side)][i] > 0:
                legal_moves.append(Move(side, i))
        return legal_moves

    @staticmethod
    def is_legal_move(board: Board, move: Move, north_moved: bool) -> bool:
        if move.index == 0:
            if move.side == Side.SOUTH:
                return False
            if not north_moved:
                return True
        return (move.index >= 1) and (move.index <= board.holes) and (board.get_seeds(move.side, move.index) > 0)

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
        if not MancalaEnv.is_legal_move(board, move, north_moved):
            raise ValueError('Move is illegal: Board: \n {} \n Move:\n {}/{} \n {}'.format(board, move.index, move.side, north_moved))

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
                and board.get_seeds(sow_side, sow_hole) \
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
        if sow_hole == 0:
            return move.side  # Last seed was placed in the store, so side moves again
        return Side.opposite(move.side)

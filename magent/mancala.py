from magent.board import Board
from magent.move import Move
from magent.side import Side
from copy import deepcopy
from typing import List


class MancalaGameState(object):
    def __init__(self):
        self._board = Board(7, 7)
        self._side_to_move = Side.SOUTH
        self._north_moved = False

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

    @staticmethod
    def clone(other_state):
        board = Board.clone(other_state.board)
        side_to_move = deepcopy(other_state.side_to_move)
        north_moved = deepcopy(other_state.north_moved)

        clone_game = MancalaGameState()
        clone_game.board = board
        clone_game.side_to_move = side_to_move
        clone_game.north_moved = north_moved
        return clone_game

    def get_legal_moves(self) -> List[Move]:
        return MancalaGameState.get_state_legal_moves(self.board, self.side_to_move, self.north_moved)

    def is_legal(self, move: Move) -> bool:
        return MancalaGameState.is_legal_move(self.board, move, self.north_moved)

    def perform_move(self, move: Move):
        self.side_to_move = MancalaGameState.make_move(self.board, move, self.north_moved)
        if move.side is Side.NORTH:
            self.north_moved = True

    def is_game_over(self) -> (bool, Side):
        return MancalaGameState.game_over(self.board)

    def get_winner(self):
        game_over, finished_side = self.is_game_over()
        if game_over is False:
            raise ValueError('This method should be called only when the game is over')

        not_finished_side = Side.opposite(finished_side)
        seeds_outside_store = 0
        for hole in

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
        if move.index is 0 and north_moved:
            return False
        return (move.index >= 1) and (move.index <= board.holes) and (board.get_seeds(move.side, move.index) != 0)

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
        if MancalaGameState.holes_empty(board, Side.SOUTH):
            return True, Side.SOUTH
        if MancalaGameState.holes_empty(board, Side.NORTH):
            return True, Side.NORTH
        return False, None

    @staticmethod
    def switch_sides(board: Board):
        for hole in range(board.holes + 1):
            board.board[0][hole], board.board[1][hole] = board.board[1][hole], board.board[0][hole]

    @staticmethod
    def make_move(board: Board, move: Move, north_moved):
        if not MancalaGameState.is_legal_move(board, move, north_moved):
            raise ValueError('Move is illegal')

        # This is a pie move
        if move.index is 0:
            MancalaGameState.switch_sides(board)
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
        if rounds is not 0:
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
        game_over, finished_side = MancalaGameState.game_over(board)
        if game_over:
            seeds = 0
            collecting_side = Side.opposite(finished_side)
            for hole in range(1, board.holes + 1):
                seeds += board.get_seeds(collecting_side, hole)
                board.set_seeds(collecting_side, hole, 0)
            board.add_seeds_to_store(collecting_side, seeds)

        # Return the side which is next to move
        if sow_hole is 0:
            return move.side  # Last seed was placed in the store, so side moves again
        return Side.opposite(move.side)

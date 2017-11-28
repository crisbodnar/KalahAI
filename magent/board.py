from magent.side import Side
from copy import deepcopy


class Board(object):
    def __init__(self, holes: int, seeds: int):
        if holes < 1:
            raise ValueError('There has to be at least one hole')
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')

        self._holes = holes

        # Place the seeds in the holes
        self.board = [[0 for _ in range(holes + 1)] for _ in range(2)]
        for hole in range(1, holes + 1):
            self.board[Side.get_index(Side.NORTH)][hole] = seeds
            self.board[Side.get_index(Side.SOUTH)][hole] = seeds

    @classmethod
    def clone(cls, original_board):
        holes = original_board.holes
        board = cls(holes, 0)
        for hole in range(1, holes + 1):
            board.board[Side.get_index(Side.NORTH)][hole] \
                = deepcopy(original_board.board[Side.get_index(Side.NORTH)][hole])
            board.board[Side.get_index(Side.SOUTH)][hole] \
                = deepcopy(original_board.board[Side.get_index(Side.SOUTH)][hole])
        return board

    @property
    def holes(self):
        return self._holes

    def get_seeds(self, side: Side, hole: int) -> int:
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number must be between 1 and number of holes')
        return self.board[Side.get_index(side)][hole]

    def set_seeds(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number must be between 1 and number of holes')
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')

        self.board[Side.get_index(side)][hole] = seeds

    def get_seeds_op(self, side: Side, hole: int):
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number must be between 1 and number of holes')
        return self.board[Side.get_index(Side.opposite(side))][hole]

    def set_seeds_op(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number must be between 1 and number of holes')
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')

        self.board[Side.get_index(Side.opposite(side))][hole] = seeds

    def add_seeds(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number must be between 1 and number of holes')
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')

        self.board[Side.get_index(side)][hole] += seeds

    def add_seeds_to_store(self, side: Side, seeds: int):
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')
        self.board[Side.get_index(side)][0] += seeds

    def set_seeds_in_store(self, side: Side, seeds: int):
        if seeds < 0:
            raise ValueError('There has to be a non-negative number of seeds')
        self.board[Side.get_index(side)][0] = seeds

    def get_seeds_in_store(self, side: Side):
        return self.board[Side.get_index(side)][0]

    def __str__(self):
        board_str = str(self.board[Side.get_index(Side.NORTH)][0]) + " --"
        for i in range(self.holes, 0, -1):
            board_str += " " + str(self.board[Side.get_index(Side.NORTH)][i])
        board_str += "\n"

        for i in range(1, self.holes + 1, 1):
            board_str += " " + str(self.board[Side.get_index(Side.SOUTH)][i])
        board_str += " --  " + str(self.board[Side.get_index(Side.SOUTH)][0]) + "\n"

        return board_str

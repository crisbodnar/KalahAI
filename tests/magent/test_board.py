import unittest
from magent.side import Side
from magent.board import Board


class TestBoard(unittest.TestCase):

    def test_from_board_works(self):
        board = Board(7, 7)
        board.set_seeds(Side.SOUTH, 5, 7)
        clone = Board.clone(board)

        # Test they are a identical
        for hole in range(1, board.holes + 1):
            self.assertEqual(board.get_seeds(Side.SOUTH, hole), clone.get_seeds(Side.SOUTH, hole))
            self.assertEqual(board.get_seeds(Side.NORTH, hole), clone.get_seeds(Side.NORTH, hole))
        self.assertEqual(board.get_seeds_in_store(Side.SOUTH), clone.get_seeds_in_store(Side.SOUTH))
        self.assertEqual(board.get_seeds_in_store(Side.NORTH), clone.get_seeds_in_store(Side.NORTH))


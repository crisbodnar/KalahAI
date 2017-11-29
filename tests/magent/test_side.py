import unittest
from magent.mancala import MancalaEnv
from magent.side import Side
from magent.move import Move


class TestSide(unittest.TestCase):

    def test_side_index_is_correct(self):
        self.assertEqual(Side.get_index(Side.NORTH), 0)
        self.assertEqual(Side.get_index(Side.SOUTH), 1)

    def test_side_opposite_is_correct(self):
        self.assertEqual(Side.opposite(Side.NORTH), Side.SOUTH)
        self.assertEqual(Side.opposite(Side.SOUTH), Side.NORTH)
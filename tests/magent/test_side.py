import unittest
from magent.side import Side


class TestSide(unittest.TestCase):

    def test_side_index_is_correct(self):
        self.assertEqual(Side.get_index(Side.NORTH), 0)
        self.assertEqual(Side.get_index(Side.SOUTH), 1)

    def test_side_opposite_is_correct(self):
        self.assertEqual(Side.opposite(Side.NORTH), Side.SOUTH)
        self.assertEqual(Side.opposite(Side.SOUTH), Side.NORTH)
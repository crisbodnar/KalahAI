import unittest
from magent.mancala import MancalaGameState
from magent.side import Side
from magent.move import Move


class TestMancalaGameState(unittest.TestCase):
    def setUp(self):
        self.game = MancalaGameState()

    def test_initial_state_is_correct(self):
        self.assertEqual(self.game.side_to_move, Side.SOUTH)
        self.assertFalse(self.game.north_moved)

        for hole in range(1, self.game.board.holes + 1):
            self.assertEqual(self.game.board.get_seeds(Side.SOUTH, hole), 7)
            self.assertEqual(self.game.board.get_seeds(Side.NORTH, hole), 7)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.SOUTH), 0)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.NORTH), 0)

    def test_cloning_immutability(self):
        clone = MancalaGameState.clone(self.game)
        self.game.perform_move(Move(Side.SOUTH, 3))

        self.assertEqual(clone.board.get_seeds(Side.SOUTH, 3), 7)
        self.assertEqual(clone.side_to_move, Side.SOUTH)
    
    def test_move_has_required_effects(self):
        self.game.perform_move(Move(Side.SOUTH, 5))
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 5), 0)
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 6), 8)
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 7), 8)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.SOUTH), 1)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 1), 8)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 2), 8)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 3), 8)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 4), 8)

        self.game.perform_move(Move(Side.NORTH, 4))
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 4), 0)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 5), 8)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 6), 8)
        self.assertEqual(self.game.board.get_seeds(Side.NORTH, 7), 8)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.NORTH), 1)
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 1), 8)
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 2), 8)
        self.assertEqual(self.game.board.get_seeds(Side.SOUTH, 3), 8)

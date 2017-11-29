import unittest
from magent.mancala import MancalaEnv
from magent.side import Side
from magent.move import Move


class TestMancalaGameState(unittest.TestCase):
    def setUp(self):
        self.game = MancalaEnv()

    def test_initial_state_is_correct(self):
        self.assertEqual(self.game.side_to_move, Side.SOUTH)
        self.assertFalse(self.game.north_moved)

        for hole in range(1, self.game.board.holes + 1):
            self.assertEqual(self.game.board.get_seeds(Side.SOUTH, hole), 7)
            self.assertEqual(self.game.board.get_seeds(Side.NORTH, hole), 7)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.SOUTH), 0)
        self.assertEqual(self.game.board.get_seeds_in_store(Side.NORTH), 0)

    def test_cloning_immutability(self):
        clone = MancalaEnv.clone(self.game)
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

    def test_game_is_over_returns_false(self):
        self.assertFalse(self.game.is_game_over())

    def test_game_is_over_returns_true(self):
        board = self.game.board
        for hole in range(1, board.holes + 1):
            board.set_seeds(Side.SOUTH, hole, 0)
        board.set_seeds_in_store(Side.SOUTH, 49)
        self.assertTrue(self.game.is_game_over())

    def test_game_returns_winner_the_player_with_most_seeds(self):
        board = self.game.board
        for hole in range(1, self.game.board.holes + 1):
            board.set_seeds(Side.SOUTH, hole, 0)
            board.set_seeds(Side.NORTH, hole, 0)
        board.set_seeds_in_store(Side.SOUTH, 23)
        board.set_seeds_in_store(Side.NORTH, 21)

        self.assertEqual(self.game.get_winner(), Side.SOUTH)

    def test_game_returns_no_winner_if_players_have_equal_number_of_seeds(self):
        board = self.game.board
        for hole in range(1, self.game.board.holes + 1):
            board.set_seeds(Side.SOUTH, hole, 0)
            board.set_seeds(Side.NORTH, hole, 0)
        board.set_seeds_in_store(Side.SOUTH, 30)
        board.set_seeds_in_store(Side.NORTH, 30)

        self.assertEqual(self.game.get_winner(), None)

    def test_is_legal_move_returns_true_for_the_pie_rule(self):
        board = self.game.board
        MancalaEnv.make_move(board, Move(Side.SOUTH, 6), False)
        self.assertTrue(MancalaEnv.is_legal_move(board, Move(Side.NORTH, 0), False))

    def test_is_legal_move_returns_true_for_the_pie_rule2(self):
        env = MancalaEnv()
        env.perform_move(Move(Side.SOUTH, 5))
        print(env.board)
        self.assertTrue(env.is_legal(Move(Side.NORTH, 0)))

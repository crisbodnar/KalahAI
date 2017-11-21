from magent.board import Board
from magent.move import Move
from magent.side import Side


# TODO(cristian): Does this have to be a class?
# seems like a collection of game rules, otherwise reformat functions to use self.board
class Mancala(object):
    @staticmethod
    # Generate a set of all legal moves given a board state and a side
    def get_legal_moves(board: Board, side: Side):
        possible_moves = []
        for i in range(1, board.holes + 1):
            if board.board[side.get_index(side)][i] > 0:
                possible_moves.append(i)
        return possible_moves

    @staticmethod
    def is_legal_move(board: Board, move: Move):
        return (move.hole <= board.holes) and (board.get_seeds(move.side, move.hole) != 0)

    @staticmethod
    def holes_empty(board: Board, side: Side):
        for hole in range(1, board.holes + 1):
            if board.get_seeds(side, hole) > 0:
                return False
        return True

    @staticmethod
    def game_over(board: Board):
        if Mancala.holes_empty(board, Side.SOUTH):
            return True, Side.SOUTH
        if Mancala.holes_empty(board, Side.NORTH):
            return True, Side.NORTH
        return False, None

    @staticmethod
    def make_move(board: Board, move: Move):
        if not Mancala.is_legal_move(board, move):
            raise ValueError('Move is illegal')

        seeds_to_sow = board.get_seeds(move.side, move.hole)
        board.set_seeds(move.side, move.hole, 0)

        holes = board.holes
        # Place seeds in all holes excepting the opponent's store
        receiving_holes = 2 * holes + 1
        # Rounds needed to sow all the seeds
        rounds = seeds_to_sow / receiving_holes
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
        sow_hole = move.hole
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
        game_over, finished_side = Mancala.game_over(board)
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

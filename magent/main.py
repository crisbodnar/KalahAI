import logging
from magent.board import Board
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
import magent.protocol.protocol as protocol
from magent.mcts.mcts import mcts_factory
from magent.side import Side
from magent.mancala import MancalaEnv

logging.basicConfig(filename='application.log', level=logging.INFO)

if __name__ == '__main__':
    mcts = mcts_factory('standard-mcts')  # configure MCTS
    our_side: Side
    board = Board(7, 7)
    state = MancalaEnv()

    try:
        while True:
            msg = protocol.read_msg()
            try:
                msg_type = protocol.get_msg_type(msg)
                if msg_type == MsgType.START:
                    state.board = board
                    first = protocol.interpret_start_msg(msg)
                    if first:
                        our_side = Side.SOUTH
                        state.side_to_move = Side.SOUTH
                        move = mcts.search(state)
                        message = protocol.create_move_msg(move.index)
                        protocol.send_msg(message)
                    else:
                        our_side = Side.NORTH

                elif msg_type == MsgType.STATE:
                    move_turn = protocol.interpret_state_msg(msg, board)
                    if move_turn.move == -1:
                        # swap movement
                        our_side = Side.opposite(our_side)
                        state.north_moved = True

                    if not move_turn.end:
                        if move_turn.again:
                            state.board = board
                            state.side_to_move = our_side
                            move = mcts.search(state)
                            # pie rule
                            if not state.north_moved and move.index == -1:
                                protocol.create_swap_msg()
                                our_side = Side.opposite(our_side)
                            else:
                                message = protocol.create_move_msg(move.index)
                                protocol.send_msg(message)

                            state.north_moved = True
                            logging.info("Our side: " + str(our_side))
                        logging.info("The board:\n" + str(board))
                elif msg_type == MsgType.END:
                    break
                else:
                    logging.warning("Not sure what I got " + str(msg_type))
            except InvalidMessageException as e:
                logging.error(str(e))
    except Exception as e:
        logging.error("Uncaught exception in main: " + str(e))
        # TODO Default to reasonable move behaviour on failure
        # protocol.send_msg("MOVE;1")

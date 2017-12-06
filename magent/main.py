import logging

from magent.move import Move
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
import magent.protocol.protocol as protocol
from magent.mcts.mcts import mcts_factory
from magent.mancala import MancalaEnv

logging.basicConfig(filename='application.log', level=logging.INFO)
# logging.basicConfig(filename='debug.log', level=logging.DEBUG)

if __name__ == '__main__':
    mcts = mcts_factory('standard-mcts')  # configure MCTS
    state = MancalaEnv()

    try:
        while True:
            msg = protocol.read_msg()
            try:
                msg_type = protocol.get_msg_type(msg)
                if msg_type == MsgType.START:
                    first = protocol.interpret_start_msg(msg)
                    if first:
                        move = mcts.search(state)
                        protocol.send_msg(protocol.create_move_msg(move.index))
                elif msg_type == MsgType.STATE:
                    move_turn = protocol.interpret_state_msg(msg)
                    state.perform_move(Move(state.side_to_move, move_turn.move))
                    if not move_turn.end:
                        if move_turn.again:
                            move = mcts.search(state)
                            # pie rule; optimal move is to swap
                            if move.index == 0:
                                protocol.send_msg(protocol.create_swap_msg())
                            else:
                                protocol.send_msg(protocol.create_move_msg(move.index))

                    logging.info("Next side: " + str(state.side_to_move))
                    logging.info("The board:\n" + str(state.board))
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

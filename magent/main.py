import logging

import tensorflow as tf

import magent.protocol.protocol as protocol
from magent.mancala import MancalaEnv
from magent.mcts.mcts import MCTSFactory
from magent.move import Move
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
from models.client import A3Client

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/tmp/kalah/main.log',
                    filemode='w')


# define a Handler which writes INFO messages or higher to the sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.DEBUG)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# # tell the handler to use this format
# console.setFormatter(formatter)
# # add the handler to the root logger
# logging.getLogger('').addHandler(console)


def main(_):
    with tf.Session() as sess:
        with tf.variable_scope("global"):
            a3client = A3Client(sess)
            mcts = MCTSFactory.alpha_mcts(a3client)

            state = MancalaEnv()
            try:
                _run_game(mcts, state)
            except Exception as e:
                logging.error("Uncaught exception in main: " + str(e))
                # TODO uncomment before release: Default to reasonable move behaviour on failure
                # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))


def _run_game(mcts, state):
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


if __name__ == '__main__':
    tf.app.run()

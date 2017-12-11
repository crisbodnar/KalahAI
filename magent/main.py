import logging
import tensorflow as tf

from magent.mcts.policies.default_policy import AlphaGoDefaultPolicy
from magent.mcts.policies.tree_policy import AlphaGoTreePolicy
from magent.move import Move
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
import magent.protocol.protocol as protocol
from magent.mcts.mcts import MCTS
from magent.mancala import MancalaEnv
from random import choice

from models.client import A3Client

logging.basicConfig(filename='application.log', level=logging.DEBUG)

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/tmp/main.log',
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


# logging.basicConfig(filename='debug.log', level=logging.DEBUG)

def main(_):
    with tf.Session() as sess:
        with tf.variable_scope("global"):
            a3client = A3Client(sess)

            mcts = MCTS(tree_policy=AlphaGoTreePolicy(a3client),
                        default_policy=AlphaGoDefaultPolicy(a3client),
                        time_sec=20)

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
                # TODO uncomment before release: Default to reasonable move behaviour on failure
                # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))


if __name__ == '__main__':
    tf.app.run()

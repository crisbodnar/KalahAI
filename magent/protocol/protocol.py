import sys
import logging
from magent.protocol.msg_type import MsgType
from magent.protocol.move_turn import MoveTurn
from magent.protocol.invalid_message_exception import InvalidMessageException


def send_msg(msg: str):
    """
        Send a message to the game engine.
        @:param msg the message
    """
    logging.info("Message to be sent: " + msg)
    print(msg, end="\n")
    sys.stdout.flush()


def read_msg() -> str:
    """
        Receives a message from the game engine. Messages are terminated by
        a '\n' character.

        @:return The message.
        @:raise IOException if there has been an I/O error.
    """

    msg = sys.stdin.readline()
    logging.info('Received: ' + msg)
    return msg


def create_move_msg(hole: int) -> str:
    """
        Creates a move message.
        @:param hole The hole to pick the seeds from.
        @:return The message as a string.
    """
    return "MOVE;" + str(hole)


def create_swap_msg() -> str:
    """
        Create a swap message.

        @:returnThe message as a string
    """
    logging.info("We swapped")
    return "SWAP"


def get_msg_type(msg: str) -> MsgType:
    if msg.startswith("START;"):
        logging.debug("Game engine sent a START command.")
        return MsgType.START
    elif msg.startswith("CHANGE;"):
        logging.debug("Game engine sent a CHANGE command.")
        return MsgType.STATE
    elif msg.startswith("END\n"):
        logging.debug("Game engine sent an END command.")
        return MsgType.END
    else:
        raise InvalidMessageException("Could not determine message type.")


def interpret_start_msg(msg: str) -> bool:
    """
        Interprets a "new_match" message. Should be called if
        getMessageType(msg) returns MsgType.START
        @:param msg The message.
        @:return "true" if this agent is the starting player (South), "false" otherwise.
        @:raises InvalidMessageException if the message is not well-formed.
    """
    if msg[-1] != '\n':
        raise InvalidMessageException("Message not terminated with 0x0A character.")

    # Message are of the form START:<POSITION> \n
    position = msg[6:-1]
    if position == "South":
        return True
    elif position == "North":
        return False
    else:
        raise InvalidMessageException("Illegal position parameter: " + position)


def interpret_state_msg(msg: str) -> MoveTurn:
    """
        Interprets a "state_change" message. Should be called if
        getMessageType(msg) returns MsgType.STATE

        @:param msg   The message.
        @:return information about the move that led to the state change and
        who's turn it is next.
        @:raises InvalidMessageException if the message is not well-formed.
    """
    move_turn = MoveTurn()
    if msg[-1] != '\n':
        raise InvalidMessageException("Message not terminated with 0x0A character.")

    msg_parts = msg.split(';', 4)

    if len(msg_parts) != 4:
        raise InvalidMessageException("Missing arguments.")
    # msgParts[0] is "CHANGE"
    # 1st argument: the move (or swap)

    if msg_parts[1] == "SWAP":
        logging.info("Opponent requested swap")
        move_turn.move = 0

    else:
        try:
            move_turn.move = int(msg_parts[1])
        except ValueError as e:
            raise InvalidMessageException("Illegal value for move parameter: ", str(e))

    # 3rd argument: who's turn
    move_turn.end = False
    if msg_parts[3] == "YOU\n":
        move_turn.again = True
    elif msg_parts[3] == "OPP\n":
        move_turn.again = False
    elif msg_parts[3] == "END\n":
        move_turn.end = True
        move_turn.again = False
    else:
        raise InvalidMessageException("Illegal value for turn parameter: " + msg_parts[3])

    logging.info("This was the move: " + str(move_turn.move))
    logging.info("Is the game over? " + str(move_turn.end))
    logging.info("Is it our turn again? " + str(move_turn.again))

    return move_turn

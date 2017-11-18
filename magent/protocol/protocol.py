from magent.board import Board
from magent.side import Side
from magent.protocol.msg_type import MsgType
from magent.protocol.move_turn import MoveTurn
from magent.protocol.invalid_message_exception import InvalidMessageException


class Protocol(object):
    """
     Creates a move message.
     @:param hole The hole to pick the seeds from.
     @:return The message as a string.
    """

    @staticmethod
    def create_move_msg(hole: int):
        return "MOVE;" + str(hole) + "\n"

    """
        Create a swap message.
        
        @:returnThe message as a string
    """

    @staticmethod
    def create_swap_msg():
        return "SWAP\n"

    @staticmethod
    def get_msg_type(msg: str):
        if msg.startswith("START;"):
            return MsgType.START
        elif msg.startswith("CHANGE;"):
            return MsgType.STATE
        elif msg.startswith("END\n"):
            return MsgType.END
        else:
            raise InvalidMessageException("Could not determine message type.")

    """
        Interprets a "new_match" message. Should be called if 
        getMessageType(msg) returns MsgType.START
        @:param msg The message.
        @:return "true" if this agent is the starting player (South), "false" otherwise.
        @:raises InvalidMessageException if the message is not well-formed.
    """

    @staticmethod
    def interpret_start_msg(msg: str):
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

    """
        Interprets a "state_change" message. Should be called if
        getMessageType(msg) returns MsgType.STATE
        
        @:param msg   The message.
        @:param board This is an output parameter. It will store the new state
                     of the Mancala board. The board has to have the right dimensions
                     (number of holes), otherwise an InvalidMessageException is
                     thrown.
        @:return information about the move that led to the state change and
        who's turn it is next.
        @:raises InvalidMessageException if the message is not well-formed.
    """

    @staticmethod
    def interpret_state_msg(msg: str, board: Board):
        move_turn = MoveTurn()
        if msg[-1] != '\n':
            raise InvalidMessageException("Message not terminated with 0x0A character.")

        msg_parts = msg.split(';', 4)

        if len(msg_parts) != 4:
            raise InvalidMessageException("Missing arguments.")
        # msgParts[0] is "CHANGE"
        # 1st argument: the move (or swap)

        if msg_parts[1] == "SWAP":
            move_turn.move = -1

        else:
            try:
                move_turn.move = int(msg_parts[1])
            except ValueError as e:
                raise InvalidMessageException("Illegal value for move parameter: ", str(e))

        # 2nd argument: the board
        board_parts = msg_parts[2].split(',', -1)

        if 2 * board.holes + 1 != len(board_parts):
            raise InvalidMessageException("Board dimensions in message ("
                                          + str(len(board_parts)) + " entries) are not as expected ("
                                          + 2 * (board.holes() + 1) + " entries).")
        try:
            for i in range(board.holes):
                # holes on the north side
                board.set_seeds(Side.NORTH, i + 1, int(board_parts[i]))
                # holes on the south side
                board.set_seeds(Side.SOUTH, i + 1, int(board_parts[i + int(board.holes) + 1]))
            # northern store
            board.add_seeds_to_store(Side.NORTH, int(board_parts[board.holes]))
            # southern store
            board.add_seeds_to_store(Side.SOUTH, int(board_parts[2 * board.holes + 1]))
        except ValueError as e:
            raise InvalidMessageException("Illegal value for seed count: " + str(e))
        except Exception as e:
            raise InvalidMessageException("Illegal value for seed count: " + str(e))

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
        return move_turn

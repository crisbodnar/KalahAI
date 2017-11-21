import sys

from magent.board import Board
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
from magent.protocol.protocol import Protocol


class Main(object):
    # Input from the game engine

    """
        Send a message to the game engine.
        @:param msg the message
    """

    @staticmethod
    def send_msg(message: str):
        print(message)
        sys.stdout.flush()

    """
        Receives a message from the game engine. Messages are terminated by 
        a '\n' character.
        
        @:return The message.
        @:raise IOException if there has been an I/O error.
    """

    @staticmethod
    def recv_msg():
        return sys.stdin.readline()


if __name__ == '__main__':
    try:
        while True:
            print()
            msg = Main.recv_msg()
            print("Received: ", msg)
            try:
                msg_type = Protocol.get_msg_type(msg)
                if msg_type == MsgType.START:
                    print("A start.")
                    first = Protocol.interpret_start_msg(msg)
                elif msg_type == MsgType.STATE:
                    print("A state.")
                    board = Board(7, 7)
                    move_turn = Protocol.interpret_state_msg(msg, board)
                    print("This was the move: ", move_turn.move)
                    print("Is the game over? ", move_turn.end)
                    if not move_turn.end:
                        print("Is it our turn again? " + str(move_turn.again))
                    print("The board:\n", board)
                elif msg_type == MsgType.END:
                    print("The end, kkthxbi")
                    break
                else:
                    print("Not sure what I got ", msg_type)
            except InvalidMessageException as e:
                print(str(e))
    except Exception as e:
        print("This shouldn't happen: " + str(e))

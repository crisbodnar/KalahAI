import argparse
from random import choice

import numpy as np

from magent.mancala import MancalaEnv
from magent.move import Move
from magent.protocol import protocol
from magent.protocol.invalid_message_exception import InvalidMessageException
from magent.protocol.msg_type import MsgType
from magent.side import Side


class Player(object):
    @staticmethod
    def get_play(state: MancalaEnv) -> Move:
        raise NotImplementedError("get_play method is not implemented")


class RandomPlayer(Player):
    @staticmethod
    def get_play(state: MancalaEnv) -> Move:
        return choice(state.get_legal_moves())


class ObservedState(object):
    """ Stores state, action_taken, side of every move opponents make"""

    def __init__(self, state: MancalaEnv, action_taken: Move):
        self.state = MancalaEnv.clone(state)
        self.action_taken = Move.clone(action_taken)


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-r', '--run-number', default=1, type=int)
parser.add_argument('-c', '--category', default="", type=str)

_checkpoint_file_path = "./saved_ckpt/observer"


def start():
    _state = MancalaEnv()
    _player = RandomPlayer()
    try:
        _run_game(_player, _state)
    except Exception as e:
        print("Uncaught exception in main: " + str(e))
        # TODO uncomment before release: Default to reasonable move behaviour on failure
        # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))


def _run_game(player: Player, state: MancalaEnv):
    our_agent_states = []
    their_agent_states = []
    both_agent_states = []
    our_side = Side.SOUTH
    while True:
        msg = protocol.read_msg()
        try:
            msg_type = protocol.get_msg_type(msg)
            if msg_type == MsgType.START:
                first = protocol.interpret_start_msg(msg)
                if first:
                    move = player.get_play(state)
                    protocol.send_msg(protocol.create_move_msg(move.index))
                else:
                    our_side = Side.NORTH
            elif msg_type == MsgType.STATE:
                move_turn = protocol.interpret_state_msg(msg)
                if move_turn.move == 0:
                    our_side = Side.opposite(our_side)

                move_to_perform = Move(state.side_to_move, move_turn.move)

                observed_state = ObservedState(state=state, action_taken=move_to_perform)
                both_agent_states.append(observed_state)
                if state.side_to_move == our_side:
                    our_agent_states.append(observed_state)
                else:
                    their_agent_states.append(observed_state)

                state.perform_move(move_to_perform)
                if not move_turn.end:
                    if move_turn.again:
                        move = player.get_play(state)
                        # pie rule; optimal move is to swap
                        if move.index == 0:
                            protocol.send_msg(protocol.create_swap_msg())
                        else:
                            protocol.send_msg(protocol.create_move_msg(move.index))

            elif msg_type == MsgType.END:
                args = parser.parse_args()
                run_id = '%06d' % args.run_number
                run_category = args.category

                _our_agent_file_path = _checkpoint_file_path + "/our-agent/" + run_category + run_id
                _their_agent_file_path = _checkpoint_file_path + "/their-agent/" + run_category + run_id
                _both_agent_file_path = _checkpoint_file_path + "/both-agent/" + run_category + run_id

                np.save(file=_our_agent_file_path, arr=np.array(our_agent_states))
                np.save(file=_their_agent_file_path, arr=np.array(their_agent_states))
                np.save(file=_both_agent_file_path, arr=np.array(both_agent_states))
                break
            else:
                print("Not sure what I got " + str(msg_type))
        except InvalidMessageException as _e:
            print(str(_e))


if __name__ == '__main__':
    start()

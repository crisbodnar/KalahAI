from enum import Enum


class MsgType(Enum):
    # message announcing the start of the game ("new_match" message)
    START = 'new_match'
    # message describing a move or a swap ("state_change" message)
    STATE = 'state_change'
    # message informing about the end of the game ("game_over" message)
    END = 'game_over'


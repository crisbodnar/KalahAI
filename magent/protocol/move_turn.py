class MoveTurn(object):
    def __init__(self):
        self._end = False
        self._again = False
        self._move = 0

    # true if game is over, false otherwise
    @property
    def end(self):
        return self._end

    # true if agent's turn, false otherwise
    @property
    def again(self):
        return self._again

    # number of the hole that characterises the move which has been made.
    # the move starts with picking seeds from the given hole.
    # -1 if the opponent has made a swap
    @property
    def move(self):
        return self._move

    @move.setter
    def move(self, value):
        self._move = value

    @again.setter
    def again(self, value):
        self._again = value

    @end.setter
    def end(self, value):
        self._end = value

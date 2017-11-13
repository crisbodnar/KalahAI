from magent.side import Side


class Move(object):
    def __init__(self, side: Side, hole: int):
        if hole < 1:
            raise ValueError('Hole number must be strictly greater than 0')

        self._side = side
        self._hole = hole

    @property
    def side(self):
        return self._side

    @property
    def hole(self):
        return self._hole

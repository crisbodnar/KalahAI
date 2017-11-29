from magent.side import Side


class Move(object):
    # Move represents a whole (if greater than 1) or the pie action if 0.
    def __init__(self, side: Side, index: int):
        if index < 0:
            raise ValueError('Move number must be non-negative')

        self._side = side
        self._index = index

    @property
    def side(self):
        return self._side

    @property
    def index(self):
        return self._index

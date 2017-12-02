from magent.side import Side


class Move(object):
    # Move represents a whole (if greater than 1) or the pie action if -1.
    def __init__(self, side: Side, index: int):
        if index != -1 and index <= 0:
            raise ValueError('Move number must be positive or -1 for pie action')

        self._side = side
        self._index = index

    @property
    def side(self) -> Side:
        return self._side

    @property
    def index(self) -> int:
        return self._index

    def __str__(self) -> str:
        return "Side: %s; Hole: %d" % (Side.side_to_str(self.side), self.index)

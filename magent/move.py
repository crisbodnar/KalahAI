from magent.side import Side


class Move(object):
    # Move represents a whole (if greater than 1) or the pie action if 0.
    def __init__(self, side: Side, index: int):
        if index < 0 or index > 7:
            raise ValueError('Move number must be strictly greater than 0 and less than 8')

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

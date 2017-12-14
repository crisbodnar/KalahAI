from magent.mancala import MancalaEnv
from magent.move import Move


class TreeSearch(object):
    def search(self, state: MancalaEnv) -> Move:
        raise NotImplementedError("search method is not implemented")

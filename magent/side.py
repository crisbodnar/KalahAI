from enum import Enum


class Side(Enum):
    NORTH = 0
    SOUTH = 1

    @staticmethod
    def get_index(side):
        if side == side.NORTH:
            return 0
        return 1

    @staticmethod
    def opposite(side):
        if side == side.NORTH:
            return side.SOUTH
        return side.NORTH

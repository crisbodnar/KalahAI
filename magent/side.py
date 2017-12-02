from enum import Enum

NORTH_INDEX = 0
SOUTH_INDEX = 1


class Side(Enum):
    NORTH = 0
    SOUTH = 1

    @staticmethod
    def get_index(side) -> int:
        if side is side.NORTH:
            return NORTH_INDEX
        return SOUTH_INDEX

    @classmethod
    def opposite(cls, side):
        if side is side.NORTH:
            return side.SOUTH
        return side.NORTH

    @staticmethod
    def side_to_str(side) -> str:
        if side is side.NORTH:
            return "North"
        return "South"

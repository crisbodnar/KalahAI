package MKAgent.heuristics;

import MKAgent.Board;
import MKAgent.Side;

public class SeedsDifference implements Heuristic {
    @Override
    public int score(Board board, Side side) {
        return board.getSeedsInStore(side) - board.getSeedsInStore(side.opposite());
    }
}



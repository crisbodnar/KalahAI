package MKAgent.heuristics;

import MKAgent.game.Board;
import MKAgent.game.Side;

public class SeedsDifference implements Heuristic {
    @Override
    public double getScore(Board board, Side side) {
        return board.getSeedsInStore(side) - board.getSeedsInStore(side.opposite());
    }
}



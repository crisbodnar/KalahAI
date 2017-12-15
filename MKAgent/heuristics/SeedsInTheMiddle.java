package MKAgent.heuristics;

import MKAgent.game.Board;
import MKAgent.game.Side;

public class SeedsInTheMiddle implements Heuristic {
    @Override
    public double getScore(Board board, Side side) {
        return getSeedsInTheMiddleOnSide(board, side) - getSeedsInTheMiddleOnSide(board, side.opposite());
    }

    private int getSeedsInTheMiddleOnSide(Board board, Side side) {
        int holes = board.getNoOfHoles();
        int seeds = 0;
        for(int hole = holes / 3; hole <= holes * 2 / 3; hole++) {
            seeds += board.getSeeds(side, hole);
        }
        return seeds;
    }
}

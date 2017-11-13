package MKAgent.heuristics;

import MKAgent.Board;
import MKAgent.Side;

public class SeedsInTheMiddle implements Heuristic {
    @Override
    public int score(Board board, Side side) {
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

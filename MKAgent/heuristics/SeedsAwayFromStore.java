package MKAgent.heuristics;

import MKAgent.Board;
import MKAgent.Side;

public class SeedsAwayFromStore implements Heuristic {
    @Override
    public int score(Board board, Side side) {
        return getSeedsAwayFromStoreOnSide(board, side) - getSeedsAwayFromStoreOnSide(board, side.opposite());
    }

    private int getSeedsAwayFromStoreOnSide(Board board, Side side) {
        int holes = board.getNoOfHoles();
        int seeds = 0;
        for(int hole = 1; hole <= holes / 3; hole++) {
            seeds += board.getSeeds(side, hole);
        }
        return seeds;
    }
}

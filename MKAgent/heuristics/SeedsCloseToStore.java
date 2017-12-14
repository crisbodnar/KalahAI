package MKAgent.heuristics;


import MKAgent.game.Board;
import MKAgent.game.Side;

public class SeedsCloseToStore implements Heuristic {
    @Override
    public int score(Board board, Side side) {
        return getSeedsCloseToStoreOnSide(board, side) - getSeedsCloseToStoreOnSide(board, side.opposite());
    }

    private int getSeedsCloseToStoreOnSide(Board board, Side side) {
        int holes = board.getNoOfHoles();
        int seeds = 0;
        for(int hole = holes - holes / 3; hole <= holes; hole++) {
            seeds += board.getSeeds(side, hole);
        }
        return seeds;
    }
}

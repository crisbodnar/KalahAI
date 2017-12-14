package MKAgent.heuristics;


import MKAgent.game.Board;
import MKAgent.game.Side;

public class Evaluation implements Heuristic {

    private Heuristic[] heuristics = {
        new SeedsAwayFromStore(),
        new SeedsCloseToStore(),
        new SeedsDifference(),
        new SeedsInTheMiddle()
    };

    @Override
    public int score(Board board, Side side) {
        int score = 0;
        for(Heuristic heuristic:heuristics) {
            score += heuristic.score(board, side);
        }
        return score;
    }
}

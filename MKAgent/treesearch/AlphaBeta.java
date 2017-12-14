package MKAgent.treesearch;

import MKAgent.game.Board;
import MKAgent.game.Side;
import MKAgent.heuristics.Evaluation;

// Our version of alphabeta
public class AlphaBeta implements TreeSearch {

    private final Evaluation evaluationFunction;

    public AlphaBeta() {
        this.evaluationFunction = new Evaluation();
    }

    @Override
    public int getBestMove(Board board, Side side) {
        return -1;
    }


    private int alpha_beta_search(Board board, int alpha, int beta, int depth) {
        return -1;
    }
}

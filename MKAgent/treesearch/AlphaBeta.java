package MKAgent.treesearch;

import MKAgent.game.Board;
import MKAgent.game.Move;
import MKAgent.game.Side;
import MKAgent.heuristics.Evaluation;

public class AlphaBeta implements TreeSearch{

    private final Evaluation evaluationFunction;

    public AlphaBeta() {
        this.evaluationFunction = new Evaluation();
    }

    @Override
    public Move getBestMove(Board board, Side side) {


        return null;
    }


    private int alpha_beta_search(Board board, int alpha, int beta, int depth) {

    }
}

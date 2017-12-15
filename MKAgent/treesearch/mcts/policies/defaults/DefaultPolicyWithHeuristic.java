package MKAgent.treesearch.mcts.policies.defaults;

import MKAgent.game.Kalah;
import MKAgent.game.Move;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;

import java.util.List;
import java.util.Random;

public class DefaultPolicyWithHeuristic implements DefaultPolicy {
    private static final Random randomiser = new Random();

    @Override
    public Kalah simulate(MonteCarloNode root) {
        MonteCarloNode node = root.clone();
        while (!node.isTerminal()) {
            List<Move> legalMoves = node.getState().getLegalMoves();
            Move legalMove = legalMoves.get(randomiser.nextInt(legalMoves.size()));
            node.getState().makeMove(legalMove);
        }
        return node.getState();
    }
}

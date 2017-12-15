package MKAgent.treesearch.mcts;

import MKAgent.game.Kalah;
import MKAgent.treesearch.TreeSearch;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;
import MKAgent.treesearch.mcts.policies.defaults.DefaultPolicy;
import MKAgent.treesearch.mcts.policies.defaults.DefaultPolicyWithHeuristic;
import MKAgent.treesearch.mcts.policies.rollouts.RolloutPolicy;
import MKAgent.treesearch.mcts.policies.rollouts.RolloutPolicyWithHeuristic;
import MKAgent.treesearch.mcts.policies.trees.TreePolicy;
import MKAgent.treesearch.mcts.policies.trees.TreePolicyWithHeuristic;

import java.util.ArrayList;

import static MKAgent.treesearch.mcts.graph.MonteCarloNodeUtils.selectRobustChild;

public class MonteCarlo implements TreeSearch {
    private TreePolicy treePolicy;
    private DefaultPolicy defaultPolicy;
    private RolloutPolicy rollOutPolicy;

    public MonteCarlo() {
        this.treePolicy = new TreePolicyWithHeuristic();
        this.defaultPolicy = new DefaultPolicyWithHeuristic();
        this.rollOutPolicy = new RolloutPolicyWithHeuristic();

    }

    @Override
    public int getBestMove(Kalah state) {
        // short circuit last move
        ArrayList<Integer> possibleMoves = state.getBoard().getPossibleMoves(state.getSideToMove());
        if (possibleMoves.size() == 1) {
            return possibleMoves.get(0);
        }

        MonteCarloNode root = new MonteCarloNode(state.clone(), null, null);
        int gamesPlayed = 0;
        // TODO replace with time based budget
        while (gamesPlayed < 75000) {
            MonteCarloNode node = treePolicy.select(root);
            Kalah finalState = defaultPolicy.simulate(node);
            rollOutPolicy.backpropagate(node, finalState);
            gamesPlayed++;
        }

        return selectRobustChild(root).getMove().getIndex();
    }
}

package MKAgent.treesearch.mcts.graph;

import java.util.Comparator;

public class MonteCarloNodeUtils {

    final private static double EXPLORATION_CONSTANT = 1.0 / Math.sqrt(2);
    final private static double ALPHA = 0.4; // Rave function alpha

    // select_best_child returns the child that maximise upper confidence interval (UCT applied to trees).
    public static MonteCarloNode selectBestChild(MonteCarloNode node) {
        return node.getChildren()
                .stream()
                .max(Comparator.comparing(child -> uctReward(node, child)))
                .get();
    }


    // select_robust_child returns the child that is most visited.
    public static MonteCarloNode selectRobustChild(MonteCarloNode node) {
        return node.getChildren()
                .stream()
                .max(Comparator.comparing(MonteCarloNode::getVisits))
                .get();
    }


    public static MonteCarloNode raveSelection(MonteCarloNode node) {
        return node.getChildren()
                .stream()
                .max(Comparator.comparing(child -> raveExploration(node, child)))
                .get();
    }


    private static double uctReward(MonteCarloNode node, MonteCarloNode child) {
        return (child.getReward() / child.getVisits())
                + (EXPLORATION_CONSTANT * Math.sqrt(2 * Math.log(node.getVisits()) / child.getVisits()));
    }

    private static double raveExploration(MonteCarloNode node, MonteCarloNode child) {
        return raveReward(child) + (EXPLORATION_CONSTANT * Math.sqrt(Math.log(node.getVisits() / child.getVisits())));
    }

    private static double raveReward(MonteCarloNode node) {
        return (1 - ALPHA) * (node.getReward() / node.getVisits()) + ALPHA * node.getHeuristicValue();
    }

}

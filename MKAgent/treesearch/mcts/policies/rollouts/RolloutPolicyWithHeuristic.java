package MKAgent.treesearch.mcts.policies.rollouts;

import MKAgent.game.Kalah;
import MKAgent.game.Side;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;

import static MKAgent.heuristics.Evaluation.computeEndGameReward;

public class RolloutPolicyWithHeuristic implements RolloutPolicy {

    @Override
    public void backpropagate(MonteCarloNode leafNode, Kalah finalState) {
        MonteCarloNode node = leafNode;
        while (node != null) {
            Side side = node.getParent() != null ?
                    node.getParent().getState().getSideToMove() : node.getState().getSideToMove();
            node.update(computeEndGameReward(finalState, side));
            node = node.getParent();
        }
    }
}

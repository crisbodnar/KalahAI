package MKAgent.treesearch.mcts.policies.rollouts;

import MKAgent.game.Kalah;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;

public interface RolloutPolicy {
    void backpropagate(MonteCarloNode leafNode, Kalah finalState);
}


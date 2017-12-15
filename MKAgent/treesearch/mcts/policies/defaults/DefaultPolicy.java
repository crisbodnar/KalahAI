package MKAgent.treesearch.mcts.policies.defaults;

import MKAgent.game.Kalah;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;

public interface DefaultPolicy {
    Kalah simulate(MonteCarloNode root);
}



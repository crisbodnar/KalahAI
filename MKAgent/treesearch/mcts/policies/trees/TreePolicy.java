package MKAgent.treesearch.mcts.policies.trees;

import MKAgent.treesearch.mcts.graph.MonteCarloNode;

public interface TreePolicy {
    MonteCarloNode select(MonteCarloNode root);
    MonteCarloNode expand(MonteCarloNode root);
}



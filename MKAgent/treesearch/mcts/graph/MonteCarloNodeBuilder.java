package MKAgent.treesearch.mcts.graph;

import MKAgent.game.Kalah;
import MKAgent.game.Move;

public class MonteCarloNodeBuilder {
    private Kalah state;
    private Move move;
    private MonteCarloNode parent;
    private int visits;
    private double reward;
    private double heuristicValue;

    public MonteCarloNodeBuilder withVisits(int visits) {
        this.visits = visits;
        return this;
    }

    public MonteCarloNodeBuilder withReward(double reward) {
        this.reward = reward;
        return this;
    }

    public MonteCarloNodeBuilder withHeuristicValue(double heuristicValue) {
        this.heuristicValue = heuristicValue;
        return this;
    }

    public MonteCarloNodeBuilder withState(Kalah state) {
        this.state = state.clone();
        return this;
    }

    public MonteCarloNodeBuilder withMove(Move move) {
        this.move = move;
        return this;
    }

    public MonteCarloNodeBuilder withParent(MonteCarloNode parent) {
        this.parent = parent;
        return this;
    }

    public MonteCarloNode build() {
        MonteCarloNode node = new MonteCarloNode(state, move, parent);
        node.setHeuristicValue(this.heuristicValue);
        node.setReward(this.reward);
        node.setVisits(this.visits);
        return node;
    }
}
package MKAgent.treesearch.mcts.graph;

import MKAgent.game.Board;
import MKAgent.game.Kalah;
import MKAgent.game.Move;

import java.util.ArrayList;
import java.util.List;

public class MonteCarloNode implements Cloneable {
    private final Kalah state;
    private final Move move;
    private MonteCarloNode parent;
    private int visits;
    private double reward;
    private double heuristicValue;


    private List<MonteCarloNode> children;
    private List<Move> unexploredMoves;


    public MonteCarloNode(Kalah state, Move move, MonteCarloNode parent) {
        this.move = move;
        this.parent = parent;
        this.state = state;
        this.visits = 0;
        this.reward = 0.0;
        this.heuristicValue = 0.0;
        this.children = new ArrayList<>();
        this.unexploredMoves = state.getLegalMoves();
    }

    @Override
    public MonteCarloNode clone() {
        return new MonteCarloNodeBuilder()
                .withMove(move)
                .withParent(parent)
                .withState(state)
                .withVisits(visits)
                .withReward(reward)
                .withHeuristicValue(heuristicValue)
                .build();
    }

    public Board getBoard() {
        return null;
    }

    public void putChild(MonteCarloNode node) {
        children.add(node);
        unexploredMoves.remove(node.getMove());
    }

    public void update(double reward) {
        this.reward += reward;
        this.visits += 1;
//        for (MonteCarloNode child : children) {
//            this.heuristicValue = Math.max(child.heuristicValue, this.heuristicValue);
//        }
    }

    @Override
    public String toString() {
        return "MonteCarloNode{" +
                "visits=" + visits +
                ", reward=" + reward +
                ", state=" + state +
                ", children=" + children.size() +
                ", move=" + move +
                ", parentSide=" + parent.getState().getSideToMove() +
                ", heuristicValue=" + heuristicValue +
                ", unexploredMoves=" + unexploredMoves +
                '}';
    }

    public boolean isTerminal() {
        return state.getLegalMoves().size() == 0;
    }

    public boolean isFullyExpanded() {
        return unexploredMoves.size() == 0;
    }

    public int getVisits() {
        return visits;
    }

    public double getReward() {
        return reward;
    }

    public Kalah getState() {
        return state;
    }

    public List<MonteCarloNode> getChildren() {
        return children;
    }

    public Move getMove() {
        return move;
    }

    public MonteCarloNode getParent() {
        return parent;
    }

    public double getHeuristicValue() {
        return heuristicValue;
    }

    public List<Move> getUnexploredMoves() {
        return unexploredMoves;
    }

    public void setVisits(int visits) {
        this.visits = visits;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public void setHeuristicValue(double heuristicValue) {
        this.heuristicValue = heuristicValue;
    }

    public void setParent(MonteCarloNode parent) {
        this.parent = parent;
    }
}



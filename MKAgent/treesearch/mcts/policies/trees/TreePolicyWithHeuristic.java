package MKAgent.treesearch.mcts.policies.trees;

import MKAgent.game.Kalah;
import MKAgent.game.Move;
import MKAgent.heuristics.Evaluation;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static MKAgent.treesearch.mcts.graph.MonteCarloNodeUtils.selectBestChild;

public class TreePolicyWithHeuristic implements TreePolicy {
    private final static Random randomiser = new Random();

    @Override
    public MonteCarloNode select(MonteCarloNode node) {
        while (!node.isTerminal()) {
            if (!node.isFullyExpanded()) {
                return expand(node);
            } else {
                node = selectBestChild(node);
            }
        }
        return node;
    }

    @Override
    public MonteCarloNode expand(MonteCarloNode root) {
        List<Move> unexploredMoves = root.getUnexploredMoves();
        Move move = unexploredMoves.get(randomiser.nextInt(unexploredMoves.size()));
        Kalah childState = root.getState().clone();
        childState.makeMove(move);
        MonteCarloNode childNode = new MonteCarloNode(childState, move, root);
        root.putChild(childNode);
//        raveValueExpansion(childNode);
        return childNode;
    }


    private void raveValueExpansion(MonteCarloNode node) {
        if (node.getUnexploredMoves().size() == 0) {
            node.setHeuristicValue(Evaluation.computeEndGameReward(node.getState(),
                    node.getParent().getState().getSideToMove()));
            return;
        }

        List<Kalah.MoveStatePair> nextMoveStatePairs = node.getState().getNextMoveStatePairs();
        double sumOfAllValues = nextMoveStatePairs.stream().mapToDouble(Kalah.MoveStatePair::getValue).sum();
        double maxOfAllValues = nextMoveStatePairs.stream().mapToDouble(Kalah.MoveStatePair::getValue).max().getAsDouble();

        double[] dists = new double[node.getState().getBoard().getNoOfHoles() + 1];


        double[] movesValue = new double[node.getState().getBoard().getNoOfHoles() + 1];
        for (int i = 0; i < node.getState().getBoard().getNoOfHoles() + 1; i++) {
            movesValue[i] = -1e80;
        }
        for (Kalah.MoveStatePair nextMoveStatePair : nextMoveStatePairs) {
            double value = nextMoveStatePair.getValue();
            if (value != 0) {
                movesValue[nextMoveStatePair.getMove().getIndex()] = value;
            }
        }

        for (int i = 0; i < node.getState().getBoard().getNoOfHoles() + 1; i++) {
            dists[i] = Math.exp(movesValue[i] - maxOfAllValues) / sumOfAllValues;
        }

        node.setHeuristicValue(Arrays.stream(dists).max().getAsDouble());
    }
}

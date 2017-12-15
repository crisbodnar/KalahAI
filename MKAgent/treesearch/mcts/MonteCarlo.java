package MKAgent.treesearch.mcts;

import MKAgent.game.Kalah;
import MKAgent.game.Move;
import MKAgent.treesearch.TreeSearch;
import MKAgent.treesearch.mcts.graph.MonteCarloNode;
import MKAgent.treesearch.mcts.policies.defaults.DefaultPolicy;
import MKAgent.treesearch.mcts.policies.defaults.DefaultPolicyWithHeuristic;
import MKAgent.treesearch.mcts.policies.rollouts.RolloutPolicy;
import MKAgent.treesearch.mcts.policies.rollouts.RolloutPolicyWithHeuristic;
import MKAgent.treesearch.mcts.policies.trees.TreePolicy;
import MKAgent.treesearch.mcts.policies.trees.TreePolicyWithHeuristic;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static MKAgent.treesearch.mcts.graph.MonteCarloNodeUtils.selectRobustChild;

public class MonteCarlo implements TreeSearch {
    private static final int CUT_OFF_VISIT_DIFF = 15000;
    private final TreePolicy treePolicy;
    private final DefaultPolicy defaultPolicy;
    private final RolloutPolicy rollOutPolicy;
    private MonteCarloNode root;

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
        if (root == null) {
            root = new MonteCarloNode(state.clone(), null, null);
        }
        int gamesPlayed = 0;
        // TODO replace with time based budget
        while (gamesPlayed < 100000) {
            // check every 500 games that we need to keep doing simulation or not
//            if (gamesPlayed % 500 == 0 && !continueSimulating()) {
//                break;
//            }
            MonteCarloNode node = treePolicy.select(root);
            Kalah finalState = defaultPolicy.simulate(node);
            rollOutPolicy.backpropagate(node, finalState);
            gamesPlayed++;
        }

        return selectRobustChild(root).getMove().getIndex();
    }

    @Override
    public void performMove(int move) {
        if (root == null) {
            return;
        }
        for (MonteCarloNode child : root.getChildren()) {
            if (child.getMove().getIndex() == move) {
                root = child;
                root.setParent(null); // cut the tree
                return;
            }
        }
        for (Move unexploredMoves : root.getUnexploredMoves()) {
            if (unexploredMoves.getIndex() == move) {
                root.getState().makeMove(move);
                root = new MonteCarloNode(root.getState(), unexploredMoves, null);
                return;
            }
        }
        throw new InvalidRootStateException("No child with the same move was found");
    }

    private boolean continueSimulating() {
        if (root.getUnexploredMoves().size() > 0 ||
                root.getChildren().size() < 2) {
            return true;
        }
        List<MonteCarloNode> sortedChildren =
                root.getChildren().parallelStream()
                        .sorted(Comparator.comparing(MonteCarloNode::getVisits).reversed())
                        .collect(Collectors.toList());

        return sortedChildren.get(0).getVisits() - CUT_OFF_VISIT_DIFF <= sortedChildren.get(1).getVisits();

    }
}

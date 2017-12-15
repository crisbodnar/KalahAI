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
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import static MKAgent.treesearch.mcts.graph.MonteCarloNodeUtils.selectRobustChild;
import static java.lang.Thread.sleep;

public class MonteCarlo implements TreeSearch, Runnable {
    private static final int CUT_OFF_VISIT_DIFF = 1000;
    private static final int MAXIMUM_NUMBER_OF_GAMES = 100000;
    private final TreePolicy treePolicy;
    private final DefaultPolicy defaultPolicy;
    private final RolloutPolicy rollOutPolicy;
    private MonteCarloNode root;
    private final Lock lock;

    public MonteCarlo() {
        this.treePolicy = new TreePolicyWithHeuristic();
        this.defaultPolicy = new DefaultPolicyWithHeuristic();
        this.rollOutPolicy = new RolloutPolicyWithHeuristic();
        this.lock = new ReentrantLock();
        this.root = new MonteCarloNode(new Kalah(7, 7), null, null);
    }

    private void search() {
        while (!root.getState().gameOver()) {
            MonteCarloNode node = treePolicy.select(root);
            Kalah finalState = defaultPolicy.simulate(node);
            lock.lock();
            rollOutPolicy.backpropagate(node, finalState);
            lock.unlock();
        }
    }

    @Override
    public int getBestMove() {
        // short circuit last move
        ArrayList<Integer> possibleMoves = root.getState().getBoard().getPossibleMoves(root.getState().getSideToMove());
        if (possibleMoves.size() == 1) {
            return possibleMoves.get(0);
        }
        while (true) {
            lock.lock();
            if (root.getVisits() < MAXIMUM_NUMBER_OF_GAMES) {
                lock.unlock();
                try {
                    sleep(5000);
                } catch (InterruptedException e) { /* Let there blackhole */ }
            } else {
                int indexToReturn = selectRobustChild(root).getMove().getIndex();
                lock.unlock();
                return indexToReturn;
            }
        }
    }


    @Override
    public void performMove(int move) {
        lock.lock();
        if (root == null) {
            lock.unlock();
            return;
        }
        for (MonteCarloNode child : root.getChildren()) {
            if (child.getMove().getIndex() == move) {
                root = child;
                root.setParent(null); // cut the tree
                lock.unlock();
                return;
            }
        }
        for (Move unexploredMoves : root.getUnexploredMoves()) {
            if (unexploredMoves.getIndex() == move) {
                root.getState().makeMove(move);
                root = new MonteCarloNode(root.getState(), unexploredMoves, null);
                lock.unlock();
                return;
            }
        }
        lock.unlock();
        throw new InvalidRootStateException("No child with the same move was found");
    }

    // TODO decide on values before plugging this back in
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

    @Override
    public void run() {
        search();
    }
}

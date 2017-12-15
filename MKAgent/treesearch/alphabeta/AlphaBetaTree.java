package MKAgent.treesearch.alphabeta;

import MKAgent.game.Board;
import MKAgent.game.Kalah;
import MKAgent.game.Move;
import MKAgent.game.Side;
import MKAgent.heuristics.Evaluation;
import MKAgent.treesearch.TreeSearch;

import java.util.ArrayList;
import java.util.Collections;
// TODO Fix to use our kalah class
public class AlphaBetaTree implements TreeSearch {
    private int currentID = 0;

    public Tree buildTree(Board board, int depth, Side ourSide) {
        Node node = new Node(null, board, currentID, 0, null, ourSide, ourSide);
        ++currentID;
        Tree tree = new Tree(node, ourSide);
        buildTreeRecurse(node, 1, depth, ourSide);
        return tree;
    }

    private void buildTreeRecurse(Node root, int currentDepth, int depth, Side ourSide) {
        if (currentDepth <= depth) {
            ArrayList<Integer> possibleMoves = root.board.getPossibleMoves(root.nextMove);

            for (Integer possibleMove : possibleMoves) {
                try {
                    Board board = root.board.clone();
                    Move move = new Move(root.nextMove, possibleMove);
                    Side nextSideAfterMove = Kalah.makeMove(board, move, true);
                    Side nextOurSide = root.ourSide;
                    if (possibleMove == 8) {
                        nextOurSide = root.ourSide.opposite();
                    }

                    Node child = new Node(root, board, currentID, currentDepth,
                            possibleMove, nextSideAfterMove, nextOurSide);
                    ++currentID;
                    root.children.add(child);
                    buildTreeRecurse(child, currentDepth + 1, depth, nextOurSide);
                } catch (Exception exception) {
                    System.err.println("Error: " + exception.getMessage());
                }
            }
        } else {
            double score = Evaluation.getScore(root.board, root.nextMove);
            if (root.nextMove != ourSide) {
                score *= -1.0D;
            }
            root.score = score;
        }

    }

    private ArrayList<Node> getBestLeafNodesRecurse(Node root, ArrayList<Node> leafNodes) {
        if (root.children == null) {
            leafNodes.add(root);
        }
        for (Node child : root.children) {
            leafNodes = getBestLeafNodesRecurse(child, leafNodes);
        }

        return leafNodes;
    }

    private ArrayList<Node> getBestNodes(Tree var0) {
        ArrayList<Node> leafChildren = new ArrayList<>();
        leafChildren = getBestLeafNodesRecurse(var0.root, leafChildren);
        Collections.sort(leafChildren);
        int leafNodes = leafChildren.size() / 100; // best n nodes proportional to how many nodes there are
        ArrayList<Node> bestNodes = new ArrayList<>();
        for (int leafNodeIndex = 0; leafNodeIndex < leafNodes; leafNodeIndex++) {
            bestNodes.add(leafChildren.get(leafNodeIndex));
        }

        return bestNodes;
    }

    private int getBestMoveWithDepth(Board board, int depth, Side ourSide) {
        try {
            Tree treeForThisNode = buildTree(board, depth, ourSide);
            ArrayList<Node> bestNodes = getBestNodes(treeForThisNode);

            for (Node bestNode : bestNodes) {
                buildTreeRecurse(bestNode, depth, depth + 5, ourSide);
            }

            return treeForThisNode.alphaBeta(treeForThisNode.root);
        } catch (OutOfMemoryError var5) {
            return getBestMoveWithDepth(board, depth - 1, ourSide);
        }
    }


    @Override
    public int getBestMove() {
        return 0;
    }

    @Override
    public void performMove(int move) {
        return; // TODO implement
    }

    @Override
    public void run() {
        return; // TODO implement
    }
}


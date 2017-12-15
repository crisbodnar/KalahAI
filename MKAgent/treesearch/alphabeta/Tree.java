package MKAgent.treesearch.alphabeta;

import MKAgent.game.Side;

public class Tree {
    public Node root;
    public Side ourSide;

    public Tree(Node root, Side ourSide) {
        this.root = root;
        this.ourSide = ourSide;
    }

    public void print() {
        this.root.print();
    }

    public int minimax(Node root) {
        Node miniMaxNode = this.minimaxMain(root);
        miniMaxNode.print();
        return this.getBestMove(miniMaxNode);
    }

    private Node minimaxMain(Node root) {
        if (root.children.isEmpty()) {
            return root;
        }
        Node[] children = new Node[root.children.size()];
        for (int childIndex = 0; childIndex < root.children.size(); childIndex++) {
            children[childIndex] = minimaxMain(root.children.get(childIndex));
        }

        return ourSide == root.nextMove ? max(children) : min(children);
    }

    private Node min(Node[] childrenOfMinNode) {
//        Arrays.stream(childrenOfMinNode).min(Comparator.comparingDouble(child -> child.getScore))
        Node minChild = childrenOfMinNode[0];
        for (int childIndex = 1; childIndex < childrenOfMinNode.length; childIndex++) {
            if (minChild != null && minChild.score != null) {
                if (childrenOfMinNode[childIndex] != null && childrenOfMinNode[childIndex].score != null
                        && childrenOfMinNode[childIndex].score < minChild.score) {
                    minChild = childrenOfMinNode[childIndex];
                }
            } else {
                minChild = childrenOfMinNode[childIndex];
            }
        }

        return minChild;
    }

    private Node max(Node[] childrenOfMaxNode) {
        Node maxNode = childrenOfMaxNode[0];

        for (int childIndex = 1; childIndex < childrenOfMaxNode.length; childIndex++) {
            if (maxNode != null && maxNode.score != null) {
                if (childrenOfMaxNode[childIndex] != null && childrenOfMaxNode[childIndex].score != null
                        && childrenOfMaxNode[childIndex].score > maxNode.score) {
                    maxNode = childrenOfMaxNode[childIndex];
                }
            } else {
                maxNode = childrenOfMaxNode[childIndex];
            }
        }

        return maxNode;
    }

    public int alphaBeta(Node root) {
        Node bestNode = this.alphaBetaMain(root);
        if (bestNode == null || bestNode.parent == null) {
            bestNode = this.minimaxMain(root);
        }

        return this.getBestMove(bestNode);
    }

    public Node alphaBetaMain(Node root) {
        Node maximisingNode = new Node(Integer.MAX_VALUE);
        Node minimisingNode = new Node(Integer.MIN_VALUE);
        Node resultNode = this.alphaBetaMax(root, maximisingNode, minimisingNode);
        return resultNode;
    }

    private Node alphaBetaMax(Node root, Node maximisingNode, Node minimisingNode) {
        if (root.children.isEmpty()) {
            return root;
        }


        for (Node child : root.children) {
            Node value;
            if (child.nextMove == root.ourSide) {
                value = alphaBetaMax(child, maximisingNode, minimisingNode);
            } else {
                value = alphaBetaMin(child, maximisingNode, minimisingNode);
            }
            if (value.score == null) {
                return minimisingNode;
            }
            if (minimisingNode.score == null) {
                return value;
            }

            if (value.score >= minimisingNode.score) {
                return minimisingNode;
            }

            if (value.score > minimisingNode.score) {
                maximisingNode = value;
            }

        }
        return maximisingNode;
    }

    private Node alphaBetaMin(Node root, Node maximisingNode, Node minimisingNode) {
        if (root.children.isEmpty()) {
            return root;
        }

        for (Node child : root.children) {
            Node value;
            if (child.nextMove == root.ourSide) {
                value = alphaBetaMax(child, maximisingNode, minimisingNode);
            } else {
                value = alphaBetaMin(child, maximisingNode, minimisingNode);
            }
            if (value.score == null) {
                return maximisingNode;
            }
            if (maximisingNode.score == null) {
                return value;
            }

            if (value.score <= maximisingNode.score) {
                return maximisingNode;
            }

            if (value.score < minimisingNode.score) {
                minimisingNode = value;
            }

        }

        return minimisingNode;
    }

    private int getBestMove(Node root) {
        Node parent = root;
        if (parent.parent == null) {
            return parent.moveFromParent;
        }
        while (parent.parent.parent != null) {
            parent = parent.parent;
        }
        return parent.moveFromParent;
    }
}

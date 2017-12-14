package MKAgent.treesearch;

import MKAgent.game.Board;
import MKAgent.game.Side;

import java.util.ArrayList;
import java.util.Comparator;

public class Node implements Comparable<Node>, Comparator<Node> {
    public ArrayList<Node> children;
    public Node parent;
    public Board board;
    public Side nextMove;
    public Double score;
    public int id;
    public int depth;
    public Integer moveFromParent;
    public Side ourSide;

    public Node(Node parent, Board board, int id, int depth, Integer moveIndexFromParent, Side side, Side ourSide) {
        this.children = new ArrayList<>();
        this.parent = parent;
        this.board = board;
        this.moveFromParent = moveIndexFromParent;
        this.id = id;
        this.depth = depth;
        this.nextMove = side;
        this.score = null;
        this.ourSide = ourSide;
    }

    public Node(double score) {
        this.score = score;
        this.children = new ArrayList<>();
    }

    public void addChild(Node child) {
        this.children.add(child);
    }

    public void print() {
        System.err.println("Node: " + this.id + " Depth: " + this.depth +
                " Score: " + this.score + " Side: " + this.nextMove);
        System.err.println("Move from parent: " + this.moveFromParent);
        if (this.parent != null) {
            System.err.println("Parent: " + this.parent.id + " Children: ");
        } else {
            System.err.println("Parent: null Children: ");
        }

        for (Node childNode : this.children) {
            System.err.println("Node: " + childNode.id + " Depth: " + childNode.depth +
                    " Score: " + childNode.score + " Side: " + childNode.nextMove);
        }

    }

    @Override
    public int compare(Node node, Node otherNode) {
        double nodeScore = node.score;
        double otherNodeScore = otherNode.score;
        if (nodeScore > otherNodeScore) {
            return -1;
        } else {
            return nodeScore < otherNodeScore ? 1 : 0;
        }
    }

    @Override
    public int compareTo(Node otherNode) {
        return compare(this, otherNode);
    }
}
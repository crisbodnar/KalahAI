package MKAgent.heuristics;

import MKAgent.Board;
import MKAgent.Side;

public interface Heuristic {
    int score(Board board, Side side);
}
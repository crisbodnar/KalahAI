package MKAgent.heuristics;

import MKAgent.game.Board;
import MKAgent.game.Side;

public interface Heuristic {
    int getScore(Board board, Side side);
}
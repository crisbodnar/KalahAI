package MKAgent.heuristics;

import MKAgent.game.Board;
import MKAgent.game.Side;

public interface Heuristic {
    double  getScore(Board board, Side side);
}
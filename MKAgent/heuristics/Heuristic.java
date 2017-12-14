package MKAgent.heuristics;

import MKAgent.game.Board;
import MKAgent.game.Side;

public interface Heuristic {
    int score(Board board, Side side);
}
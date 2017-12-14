package MKAgent.treesearch;

import MKAgent.game.Board;
import MKAgent.game.Side;

public interface TreeSearch {

    int getBestMove(Board board, Side side);
}

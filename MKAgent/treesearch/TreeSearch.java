package MKAgent.treesearch;

import MKAgent.game.Board;
import MKAgent.game.Move;
import MKAgent.game.Side;

public interface TreeSearch {

    Move getBestMove(Board board, Side side);
}

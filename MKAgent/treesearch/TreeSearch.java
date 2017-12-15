package MKAgent.treesearch;

import MKAgent.game.Kalah;

public interface TreeSearch {

    int getBestMove(Kalah kalah);
    void performMove(int move);
}

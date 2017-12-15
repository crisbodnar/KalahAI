package MKAgent.treesearch;

public interface TreeSearch extends Runnable{
    int getBestMove();
    void performMove(int move);
}

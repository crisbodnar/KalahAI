package MKAgent.heuristics;


import MKAgent.game.Board;
import MKAgent.game.Kalah;
import MKAgent.game.Side;

public class Evaluation {

    public static double getScore(Board board, Side side) {
        double reward = 0.0;
        double thisSideSeeds = board.getSeedsInStore(side);
        double oppSeeds = board.getSeedsInStore(side.opposite());
        if ((thisSideSeeds != 0.0 || oppSeeds != 0.0) && thisSideSeeds != oppSeeds) {
            double var8;
            double var10;
            if (thisSideSeeds > oppSeeds) {
                var8 = thisSideSeeds;
                var10 = oppSeeds;
            } else {
                var8 = oppSeeds;
                var10 = thisSideSeeds;
            }

            reward = (1.0D / var8 * (var8 - var10) + 1.0D) * var8;
            if (oppSeeds > thisSideSeeds) {
                reward *= -1.0D;
            }
        }

        // capture move
        for (int holeIndex = 1; holeIndex <= board.getNoOfHoles(); holeIndex++) {
            if (board.getSeeds(side, holeIndex) == 0 && isSeedable(board, side, holeIndex)) {
                reward += (board.getSeedsOp(side, holeIndex) / 2.0);
            }
            if (board.getNoOfHoles() - holeIndex + 1 == board.getSeeds(side, holeIndex)) {
                reward++;
            }
        }

        int holeIndex = 0;

        int var9;
        for (var9 = 1; var9 <= board.getNoOfHoles(); ++var9) {
            holeIndex += board.getSeeds(side, var9);
        }

        var9 = 0;

        int var13;
        for (var13 = 1; var13 <= board.getNoOfHoles(); ++var13) {
            var9 += board.getSeeds(side.opposite(), var13);
        }

        var13 = holeIndex - var9;
        reward += (var13 / 2.0);

        for (int var11 = 1; var11 <= board.getNoOfHoles(); ++var11) {
            if (board.getSeeds(side.opposite(), var11) == 0 && isSeedable(board, side.opposite(), var11)) {
                reward -= (double) (board.getSeedsOp(side.opposite(), var11) / 2);
            }
        }

        return reward;
    }

    private static boolean isSeedable(Board board, Side side, int var2) {
        boolean var3 = false;

        for (int var4 = var2 - 1; var4 > 0; --var4) {
            if (var2 - var4 == board.getSeeds(side, var4)) {
                var3 = true;
                break;
            }
        }

        return var3;
    }


    public static int computeEndGameReward(Kalah finalState, Side side) {
        if (!finalState.gameOver()) {
            throw new EarlyAccessToEndResultException("Access to end results before game finish");
        }
        Board finalStateBoard = finalState.getBoard();
        // win = 1, lose = -1, tie ==0
        return Integer.compare(finalStateBoard.getSeedsInStore(side), finalStateBoard.getSeedsInStore(side.opposite()));
    }
}

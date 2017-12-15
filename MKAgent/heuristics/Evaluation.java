package MKAgent.heuristics;


import MKAgent.game.Board;
import MKAgent.game.Kalah;
import MKAgent.game.Side;

import java.util.ArrayList;
import java.util.List;

/**
 * This class contains a set of heuristics which are used for Mancala board evaluations.
 */

public class Evaluation {

    /**
     * Evaluates a node using a weighted combination of heuristics
     *
     * @param state The root node to be evaluated
     */
    public static int evaluateState(Kalah state) {
        Side ourSide = state.getOurSide();
        // Weights of heuristics
        int weight1 = 4;
        int weight2 = 2;
        int weight3 = 18;
        int weight4 = 1;
        int weight5 = 1000;
        int weight6 = 1;
        int weight7 = 2;

        return +storeDiff(state, ourSide) * weight1
                - maximumVulnerableSeeds(state, ourSide) / weight2
                + clusterTowardsStore(state, ourSide) / weight3
                + extraTurnsScore(state, ourSide) / weight4
                + ownOverHalfOfStones(state, ourSide) * weight5
                - ownOverHalfOfStones(state, ourSide.opposite()) * weight5
                + differenceOfSeeds(state, ourSide) / weight6
                + maximumVulnerableSeeds(state, ourSide.opposite()) / weight7;
    }

    /**
     * Computes the seeds difference between the two sides
     *
     * @param state The board to be checked
     * @param side  The side from which it is subtracted
     * @return The seeds difference
     */
    private static int differenceOfSeeds(Kalah state, Side side) {
        Board board = state.getBoard();
        int n1 = 0;
        int n2 = 0;
        for (int i = 1; i <= board.getNoOfHoles(); i++) {
            n1 += board.getSeeds(side, i);
            n2 += board.getSeeds(side.opposite(), i);
        }
        return n1 - n2;
    }

    /**
     * Computes the number of moves which can generate extra turns
     *
     * @param state The board to be checked
     * @param side  The side to be checked
     * @return The number of moves.
     */
    private static int extraTurnsScore(Kalah state, Side side) {
        Board board = state.getBoard();
        int score = 0;
        for (int i = 1; i <= board.getNoOfHoles(); i++) {
            int seeds = board.getSeeds(side, i);
            if (seeds == board.getNoOfHoles() + 1 - i) {
                score += 1;
            }
        }
        return score;
    }

    /**
     * Computes a score which gives more weight to seeds close to the store
     *
     * @param state Board to be checked
     * @param side  Side to be checked
     * @return The score
     */
    private static int clusterTowardsStore(Kalah state, Side side) {
        Board board = state.getBoard();
        int n = 0;
        for (int i = 1; i <= board.getNoOfHoles(); ++i) {
            n += board.getSeeds(side, i) * (i - 1);
        }
        return n;
    }

    /**
     * Returns 1 if more than half of the seeds are in the store
     *
     * @param state Board to be checked
     * @param side  The side of the store
     * @return 1 or 0
     */
    private static int ownOverHalfOfStones(Kalah state, Side side) {
        Board board = state.getBoard();
        int inStore = board.getSeedsInStore(side);
        if (inStore > 49) {
            return 1;
        }
        return 0;
    }


    /**
     * Calculates the seeds difference between stores
     */
    private static int storeDiff(Kalah state, Side side) {
        Board board = state.getBoard();
        return board.getSeedsInStore(side) - board.getSeedsInStore(side.opposite());
    }


    /**
     * Finds the maximum number of seeds from the supplied side which can be captured by the opponent next turn.
     *
     * @param state   The board to be analysed
     * @param ourSide The side with the seeds which can be captured
     * @return the number of seeds on the supplied side which can be captured
     */
    private static int maximumVulnerableSeeds(Kalah state, Side ourSide) {
        Board board = state.getBoard();
        int fullRoundCaptures = 0;

        // Check for captures which make a full round of the table
        for (int i = 1; i <= board.getNoOfHoles(); i++) {
            if (board.getSeeds(ourSide.opposite(), i) == 2 * board.getNoOfHoles() + 1) {
                fullRoundCaptures = Math.max(fullRoundCaptures, board.getSeedsOp(ourSide.opposite(), i) + 1);
            }
        }

        int captureFromLeft = 0, captureFromRight = 0;
        List<Integer> vulnerableSeeds = new ArrayList<>();

        // Finds those holes where the opponent has 0 but on the opposite side we have seeds (so they could be captured)
        for (int i = 1; i <= board.getNoOfHoles(); i++) {
            if (board.getSeeds(ourSide.opposite(), i) == 0 && board.getSeedsOp(ourSide.opposite(), i) != 0)
                vulnerableSeeds.add(i);
        }

        for (int index : vulnerableSeeds) {
            for (int i = 1; i < index; i++) {
                if (board.getSeeds(ourSide.opposite(), i) == index - i)
                    captureFromLeft = Math.max(captureFromLeft, board.getSeedsOp(ourSide.opposite(), i) + 1);
            }

            for (int i = index + 1; i <= board.getNoOfHoles(); i++) {
                if (board.getSeeds(ourSide.opposite(), i) == 2 * board.getNoOfHoles() + 1 - (i - index)) {
                    captureFromRight = Math.max(captureFromRight, board.getSeedsOp(ourSide.opposite(), i) + 1);
                }
            }
        }
        return Math.max(fullRoundCaptures, Math.max(captureFromLeft, captureFromRight));
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
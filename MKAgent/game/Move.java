package MKAgent.game;

import java.util.Objects;

/**
 * Represents a move (not a turn) in the Kalah game.
 */
public class Move {
    /**
     * The side of the board the player making the move is playing on.
     */
    private final Side side;
    /**
     * The index from which seeds are picked at the beginning of the move and
     * distributed. It has to be >= 1.
     */
    private final int index;


    /**
     * @param side  The side of the board the player making the move is playing
     *              on.
     * @param index The index from which seeds are picked at the beginning of
     *              the move and distributed. It has to be >= 1.
     * @throws IllegalArgumentException if the index number is not >= 1.
     */
    public Move(Side side, int index) throws IllegalArgumentException {
        if (index < 0)
            throw new IllegalArgumentException("Hole numbers must be >= 1, but " + index + " was given.");
        this.side = side;
        this.index = index;
    }

    /**
     * @return The side of the board the player making the move is playing on.
     */
    public Side getSide() {
        return side;
    }

    /**
     * @return The index from which seeds are picked at the beginning of the
     * move and distributed. It will be >= 1.
     */
    public int getIndex() {
        return index;
    }

    @Override
    public String toString() {
        return "Move{" +
                "side=" + side +
                ", index=" + index +
                '}';
    }

    @Override
    public boolean equals(Object anotherMove) {
        if (this == anotherMove) return true;
        if (anotherMove == null || getClass() != anotherMove.getClass()) return false;
        Move move = (Move) anotherMove;
        return index == move.index &&
                side == move.side;
    }

    @Override
    public int hashCode() {

        return Objects.hash(side, index);
    }
}

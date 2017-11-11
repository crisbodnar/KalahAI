package MKAgent;
import java.util.Observable;

/**
 * Representation of the Kalah board.<BR><BR>
 * The board has two sides: "North" and
 * "South". On each side there is a number of linearly arranged "holes" (the
 * same number on each side) and a "store" for each side. Holes are numbered
 * per side, starting with 1 on "the left" (i.e. furthest away from the
 * player's store, the numbers increase in playing direction).
 * <BR>
 * Initially, there is the same number of "seeds" in each hole.
 */
public class Board extends Observable implements Cloneable
{
	/**
	 * @see #board
	 */
	private static final int NORTH_ROW = 0;
	/**
	 * @see #board
	 */
	private static final int SOUTH_ROW = 1;

	/**
	 * The number of holes per side (must be >= 1).
	 */
	private final int holes;

	/**
	 * The board data. The first dimension of the array is 2, the second one
	 * is the number of holes per side plus one. The data for the North side
	 * is stored in board[NORTH_ROW][*], the data for the South side in
	 * board[SOUTH_ROW][*]. The number of seeds in hole number n (of one side)
	 * is stored in board[...][n], the number of seeds in the store (of one
	 * side) is stored in board[...][0].
	 */
	private int[][] board;

    /**
     * @param side A side of the board.
     * @return The index of side "side" for the first dimension of "board".
     */
    private static int indexOfSide (Side side)
    {
    	switch (side)
    	{
    		case NORTH: return NORTH_ROW;
    		case SOUTH: return SOUTH_ROW;
    		default: return -1;  // should never get here
    	}
    }


    /**
     * Creates a new board.
     * 
     * @param holes The number of holes per side (must be >= 1).
     * @param seeds The initial number of seeds per hole (must be >= 0). The
     *        stores are empty initially.
     * @throws IllegalArgumentException if any of the arguments is outside of
     *         the valid range.
     */
    public Board (int holes, int seeds) throws IllegalArgumentException
    {
    	if (holes < 1)
    		throw new IllegalArgumentException("There has to be at least one hole, but " + holes + " were requested.");
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	this.holes = holes;
    	board = new int[2][holes+1]; // WARNING: potential integer overflow here!

    	for (int i=1; i <= holes; i++)
    	{
    		board[NORTH_ROW][i] = seeds;
    		board[SOUTH_ROW][i] = seeds;
    	}
    }
    
	/**
     * Creates a new board as the copy of a given one. Both copies can then be
     * altered independently.
     *
     * @param original The board to copy.
     * @see #clone()
     */
    public Board (Board original)
    {
    	holes = original.holes;
    	board = new int[2][holes+1];

    	for (int i=0; i <= holes; i++)
    	{
    		board[NORTH_ROW][i] = original.board[NORTH_ROW][i];
    		board[SOUTH_ROW][i] = original.board[SOUTH_ROW][i];
    	}
    }

    /**
     * Creates a copy of the current board. Both copies can then be altered
     * independently.
     *  
     * @see java.lang.Object#clone()
     * @see #Board(Board)
     */
    @Override
	public Board clone() throws CloneNotSupportedException
	{
    	return new Board(this);
	}

    /**
     * @return The number of holes per side (will be >= 1).
     */
    public int getNoOfHoles (  )
    {
		return holes;
    }

    /**
     * Get the number of seeds in a hole.
     * @param side The side the hole is located on.
     * @param hole The number of the hole.
     * @return The number of seeds in hole "hole" on side "side".
     * @throws IllegalArgumentException if the hole number is invalid.
     */
    public int getSeeds (Side side, int hole) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + (board[NORTH_ROW].length - 1) + " but was " + hole + ".");

    	return board[indexOfSide(side)][hole];
    }

    /**
     * Sets the number of seeds in a hole.
     * @param side The side the hole is located on.
     * @param hole The number of the hole.
     * @param seeds The number of seeds that shall be in the hole afterwards (>= 0).
     * @throws IllegalArgumentException if any of the arguments is outside of
     *         the valid range.
     */
    public void setSeeds (Side side, int hole, int seeds) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + (board[NORTH_ROW].length - 1) + " but was " + hole + ".");
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[indexOfSide(side)][hole] = seeds;
    	this.setChanged();
    }

    /**
     * Adds seeds to a hole.
     * @param side The side the hole is located on.
     * @param hole The number of the hole.
     * @param seeds The number (>= 0) of seeds to put into (add to) the hole.
     * @throws IllegalArgumentException if any of the arguments is outside of
     *         the valid range.
     */
    public void addSeeds (Side side, int hole, int seeds) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + (board[NORTH_ROW].length - 1) + " but was " + hole + ".");
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[indexOfSide(side)][hole] += seeds;
    	this.setChanged();
    }

    /**
     * Get the number of seeds in a hole opposite to a given one.
     * @param side The side the given hole is located on.
     * @param hole The number of the given hole.
     * @return The number of seeds in the hole opposite to hole "hole" on
     *         side "side".
     * @throws IllegalArgumentException if the hole number is invalid.
     */
    public int getSeedsOp (Side side, int hole) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + holes + " but was " + hole + ".");

    	return board[1-indexOfSide(side)][holes+1-hole];
    }
 
    /**
     * Sets the number of seeds in a hole opposite to a given one.
     * @param side The side the given hole is located on.
     * @param hole The number of the given hole.
     * @param seeds The number of seeds that shall be in the hole opposite to
     *        hole "hole" on side "side" afterwards (>= 0).
     * @throws IllegalArgumentException if any of the arguments is outside of
     *         the valid range.
     */
    public void setSeedsOp (Side side, int hole, int seeds) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + (board[NORTH_ROW].length - 1) + " but was " + hole + ".");
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[1-indexOfSide(side)][holes+1-hole] = seeds;
    	this.setChanged();
    }

    /**
     * Adds seeds to a hole opposite to a given one.
     * @param side The side the given hole is located on.
     * @param hole The number of the given hole.
     * @param seeds The number (>= 0) of seeds to put into (add to) the hole opposite to
     *        hole "hole" on side "side" afterwards (>= 0).
     * @throws IllegalArgumentException if any of the arguments is outside of
     *         the valid range.
     */
    public void addSeedsOp (Side side, int hole, int seeds) throws IllegalArgumentException
    {
    	if (hole < 1 || hole > holes)
    		throw new IllegalArgumentException("Hole number must be between 1 and " + (board[NORTH_ROW].length - 1) + " but was " + hole + ".");
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[1-indexOfSide(side)][holes+1-hole] += seeds;
    	this.setChanged();
    }

    /**
     * Get the number of seeds in a store.
     * @param side The side the store is located on.
     * @return The number of seeds in the store.
     */
    public int getSeedsInStore (Side side)
    {
		return board[indexOfSide(side)][0];
    }

    /**
     * Sets the number of seeds in a store.
     * @param side The side the store is located on.
     * @param seeds The number of seeds that shall be in the store afterwards (>= 0).
     * @throws IllegalArgumentException if the number of seeds is invalid.
     */
    public void setSeedsInStore (Side side, int seeds) throws IllegalArgumentException
    {
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[indexOfSide(side)][0] = seeds;
    	this.setChanged();
    }

    /**
     * Adds seeds to a store.
     * @param side The side the store is located on.
     * @param seeds The number (>= 0) of seeds to put into (add to) the store.
     * @throws IllegalArgumentException if the number of seeds is invalid.
     */
    public void addSeedsToStore (Side side, int seeds) throws IllegalArgumentException
    {
    	if (seeds < 0)
    		throw new IllegalArgumentException("There has to be a non-negative number of seeds, but " + seeds + " were requested.");

    	board[indexOfSide(side)][0] += seeds;
    	this.setChanged();
    }

	@Override
	public String toString()
	{
		StringBuilder boardString = new StringBuilder();

		boardString.append(board[NORTH_ROW][0] + "  --");
		for (int i=holes; i >= 1; i--)
			boardString.append("  " + board[NORTH_ROW][i]);
		boardString.append("\n");
		for (int i=1; i <= holes; i++)
			boardString.append(board[SOUTH_ROW][i] + "  ");
		boardString.append("--  " + board[SOUTH_ROW][0] + "\n");

		return boardString.toString();
	}
}


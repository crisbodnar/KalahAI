package MKAgent;
/**
 * This class deals with moves on a Kalah board.
 */
public class Kalah
{
    /**
     * The board to play on.
     */
    private final Board board;

    /**
     * @param board The board to play on.
     * @throws NullPointerException if "board" is null.
     */
    public Kalah (Board board) throws NullPointerException
    {
    	if (board == null)
    		throw new NullPointerException();
    	this.board = board;
    }

    /**
     * @return The board this object operates on.
     */
    public Board getBoard ()
    {
		return board;
    }

    /**
     * Checks whether a given move is legal on the underlying board. The move
     * is not actually made.
     * @param move The move to check.
     * @return true if the move is legal, false if not.
     */
    public boolean isLegalMove (Move move)
    {
    	return isLegalMove(board, move);
    }

    /**
     * Performs a move on the underlying board. The move must be legal. If
     * the move terminates the game, the remaining seeds of the opponent are
     * collected into their store as well (so that all holes are empty).<BR>
     * The "notifyObservers()" method of the board is called with the "move"
     * as argument.
     * 
     * @param move The move to make.
     * @return The side who's turn it is after the move. Arbitrary if the
     *         game is over.
     * @see #isLegalMove(Move)
     * @see #gameOver()
     * @see java.util.Observable#notifyObservers(Object)
     */
    public Side makeMove (Move move)
    {
    	return makeMove(board, move);
    }

    /**
     * Checks whether the game is over (based on the board).
     * @return "true" if the game is over, "false" otherwise.
     */
    public boolean gameOver()
    {
    	return gameOver(board);
    }

    /**
     * Checks whether a given move is legal on a given board. The move
     * is not actually made.
     * @param board The board to check the move for.
     * @param move The move to check.
     * @return true if the move is legal, false if not.
     */
    public static boolean isLegalMove (Board board, Move move)
    {
    	// check if the hole is existent and non-empty:
    	return (move.getHole() <= board.getNoOfHoles())
    	       && (board.getSeeds(move.getSide(), move.getHole()) != 0);
    }

    /**
     * Performs a move on a given board. The move must be legal. If
     * the move terminates the game, the remaining seeds of the opponent are
     * collected into their store as well (so that all holes are empty).<BR>
     * The "notifyObservers()" method of the board is called with the "move"
     * as argument.
     * 
     * @param board The board to make the move on.
     * @param move The move to make.
     * @return The side who's turn it is after the move. Arbitrary if the
     *         game is over.
     * @see #isLegalMove(Board, Move)
     * @see #gameOver(Board)
     * @see java.util.Observable#notifyObservers(Object)
     */
    public static Side makeMove (Board board, Move move)
    {
		/* from the documentation:
		  "1. The counters are lifted from this hole and sown in anti-clockwise direction, starting
		      with the next hole. The player's own kalahah is included in the sowing, but the
		      opponent's kalahah is skipped.
		   2. outcome:
		    	1. if the last counter is put into the player's kalahah, the player is allowed to
		    	   move again (such a move is called a Kalah-move);
		    	2. if the last counter is put in an empty hole on the player's side of the board
		    	   and the opposite hole is non-empty,
		    	   a capture takes place: all stones in the opposite opponents pit and the last
		    	   stone of the sowing are put into the player's store and the turn is over;
		    	3. if the last counter is put anywhere else, the turn is over directly.
		   3. game end:
		    	The game ends whenever a move leaves no counters on one player's side, in
		    	which case the other player captures all remaining counters. The player who
		    	collects the most counters is the winner."
		*/


    	// pick seeds:
    	int seedsToSow = board.getSeeds(move.getSide(), move.getHole());
    	board.setSeeds(move.getSide(), move.getHole(), 0);

    	int holes = board.getNoOfHoles();
    	int receivingPits = 2*holes + 1;  // sow into: all holes + 1 store
    	int rounds = seedsToSow / receivingPits;  // sowing rounds
    	int extra = seedsToSow % receivingPits;  // seeds for the last partial round
    	/* the first "extra" number of holes get "rounds"+1 seeds, the
    	   remaining ones get "rounds" seeds */

    	// sow the seeds of the full rounds (if any):
    	if (rounds != 0)
    	{
    		for (int hole = 1; hole <= holes; hole++)
    		{
        		board.addSeeds(Side.NORTH, hole, rounds);
        		board.addSeeds(Side.SOUTH, hole, rounds);
    		}
    		board.addSeedsToStore(move.getSide(), rounds);
    	}

    	// sow the extra seeds (last round):
    	Side sowSide = move.getSide();
    	int sowHole = move.getHole();  // 0 means store
    	for (; extra > 0; extra--)
    	{
    		// go to next pit:
    		sowHole++;
    		if (sowHole == 1)  // last pit was a store
    			sowSide = sowSide.opposite();
    		if (sowHole > holes)
    		{
    			if (sowSide == move.getSide())
    			{
    				sowHole = 0;  // sow to the store now
    	    		board.addSeedsToStore(sowSide, 1);
    	    		continue;
    			}
    			else
    			{
        			sowSide = sowSide.opposite();
    				sowHole = 1;
    			}
    		}
    		// sow to hole:
    		board.addSeeds(sowSide, sowHole, 1);
    	}

    	// capture:
    	if ( (sowSide == move.getSide())  // last seed was sown on the moving player's side ...
    		 && (sowHole > 0)  // ... not into the store ...
    		 && (board.getSeeds(sowSide, sowHole) == 1)  // ... but into an empty hole (so now there's 1 seed) ...
    		 && (board.getSeedsOp(sowSide, sowHole) > 0) )  // ... and the opposite hole is non-empty
    	{
    		board.addSeedsToStore(move.getSide(), 1 + board.getSeedsOp(move.getSide(), sowHole));
    		board.setSeeds(move.getSide(), sowHole, 0);
    		board.setSeedsOp(move.getSide(), sowHole, 0);
    	}

    	// game over?
		Side finishedSide = null;
    	if (holesEmpty(board, move.getSide()))
    		finishedSide = move.getSide();
    	else if (holesEmpty(board, move.getSide().opposite()))
    		finishedSide = move.getSide().opposite();
    		/* note: it is possible that both sides are finished, but then
    		   there are no seeds to collect anyway */
    	if (finishedSide != null)
    	{
    		// collect the remaining seeds:
    		int seeds = 0;
    		Side collectingSide = finishedSide.opposite();
    		for (int hole = 1; hole <= holes; hole++)
    		{
    			seeds += board.getSeeds(collectingSide, hole);
    			board.setSeeds(collectingSide, hole, 0);
    		}
			board.addSeedsToStore(collectingSide, seeds);
    	}

    	board.notifyObservers(move);

    	// who's turn is it?
    	if (sowHole == 0)  // the store (implies (sowSide == move.getSide()))
    		return move.getSide();  // move again
    	else
    		return move.getSide().opposite();
    }

    /**
     * Checks whether all holes on a given side are empty.
     * @param board The board to check.
     * @param side The side to check.
     * @return "true" iff all holes on side "side" are empty.
     */
    protected static boolean holesEmpty (Board board, Side side)
    {
    	for (int hole = 1; hole <= board.getNoOfHoles(); hole++)
    		if (board.getSeeds(side, hole) != 0)
    			return false;
		return true;
    }

    /**
     * Checks whether the game is over (based on the board).
     * @param board The board to check the game state for.
     * @return "true" if the game is over, "false" otherwise.
     */
    public static boolean gameOver (Board board)
    {
    	// The game is over if one of the agents can't make another move.

    	return holesEmpty(board, Side.NORTH) || holesEmpty(board, Side.SOUTH);
    }
}


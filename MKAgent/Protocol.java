package MKAgent;
/**
 * Creates messages to be sent and interprets messages received.
 */
public class Protocol
{
	/**
	 * An object of this type is returned by interpretStateMsg().
	 * @see Protocol#interpretStateMsg(String, Board)
	 */
	public static class MoveTurn
	{
		/**
		 * "true" if the game is over, "false" otherwise.
		 */
		public boolean end;
		/**
		 * "true" if it's this agent's turn again, "false" otherwise.
		 */
		public boolean again;
		/**
		 * The number of the hole that characterises the move which has been
		 * made (the move starts with picking the seeds from the given hole)
		 * or -1 if the opponent has made a swap.
		 */
		public int move;
	}


    /**
     * Creates a "move" message.
     * @param hole The hole to pick the seeds from.
	 * @return The message as a string.
     */
    public static String createMoveMsg (int hole)
    {
		return "MOVE;" + hole + "\n";
    }

    /**
     * Creates a "swap" message.
	 * @return The message as a string.
     */
    public static String createSwapMsg ()
    {
		return "SWAP\n";
    }

    /**
     * Determines the type of a message received from the game engine. This
     * method does not check whether the message is well-formed.
     * @param msg The message.
     * @return The message type.
     * @throws InvalidMessageException if the message type cannot be
     *         determined.
     */
    public static MsgType getMessageType (String msg) throws InvalidMessageException
    {
    	if (msg.startsWith("START;"))
    		return MsgType.START;
    	else if (msg.startsWith("CHANGE;"))
    		return MsgType.STATE;
    	else if (msg.equals("END\n"))
    		return MsgType.END;
    	else
    		throw new InvalidMessageException("Could not determine message type.");
    }

    /**
     * Interprets a "new_match" message. Should be called if
     * getMessageType(msg) returns MsgType.START
     * @param msg The message.
     * @return "true" if this agent is the starting player (South), "false"
     *         otherwise.
     * @throws InvalidMessageException if the message is not well-formed.
     * @see #getMessageType(String)
     */
    public static boolean interpretStartMsg (String msg) throws InvalidMessageException
    {
    	if (msg.charAt(msg.length()-1) != '\n')
    		throw new InvalidMessageException("Message not terminated with 0x0A character.");

    	String position = msg.substring(6, msg.length()-1);
    	if (position.equals("South"))
    		return true;  // South starts the game
    	else if (position.equals("North"))
    		return false;
    	else
    		throw new InvalidMessageException("Illegal position parameter: " + position);
    }

    /**
     * Interprets a "state_change" message. Should be called if
     * getMessageType(msg) returns MsgType.STATE 
     * @param msg The message.
     * @param board This is an output parameter. It will store the new state
     *        of the Kalah board. The board has to have the right dimensions
     *        (number of holes), otherwise an InvalidMessageException is
     *        thrown.
     * @return information about the move that led to the state change and
     *         who's turn it is next.
     * @throws InvalidMessageException if the message is not well-formed.
     * @see #getMessageType(String)
     */
    public static MoveTurn interpretStateMsg (String msg, Board board) throws InvalidMessageException
    {
    	MoveTurn moveTurn = new MoveTurn();

    	if (msg.charAt(msg.length()-1) != '\n')
    		throw new InvalidMessageException("Message not terminated with 0x0A character.");

    	String[] msgParts = msg.split(";", 4);
    	if (msgParts.length != 4)
    		throw new InvalidMessageException("Missing arguments.");

    	// msgParts[0] is "CHANGE"

    	// 1st argument: the move (or swap)
    	if (msgParts[1].equals("SWAP"))
    		moveTurn.move = -1;
    	else
    	{
        	try
        	{
        		moveTurn.move = Integer.parseInt(msgParts[1]);
        	}
        	catch (NumberFormatException e)
        	{
        		throw new InvalidMessageException("Illegal value for move parameter: " + e.getMessage());
        	}
    	}

    	// 2nd argument: the board
    	String[] boardParts = msgParts[2].split(",", -1);
    	/*if (boardParts.length % 2 != 0)
    		throw new InvalidMessageException("Malformed board: odd number of entries.");*/
    	if (2*(board.getNoOfHoles()+1) != boardParts.length)
    		throw new InvalidMessageException("Board dimensions in message ("
    				+ boardParts.length + " entries) are not as expected ("
    				+ 2*(board.getNoOfHoles()+1) + " entries).");
    	try
    	{
    		// holes on the north side:
    		for (int i = 0; i < board.getNoOfHoles(); i++)
    			board.setSeeds(Side.NORTH, i+1, Integer.parseInt(boardParts[i]));
    		// northern store:
			board.setSeedsInStore(Side.NORTH, Integer.parseInt(boardParts[board.getNoOfHoles()]));
			// holes on the south side:
       		for (int i = 0; i < board.getNoOfHoles(); i++)
    			board.setSeeds(Side.SOUTH, i+1, Integer.parseInt(boardParts[i+board.getNoOfHoles()+1]));
    		// southern store:
			board.setSeedsInStore(Side.SOUTH, Integer.parseInt(boardParts[2*board.getNoOfHoles()+1]));
    	}
    	catch (NumberFormatException e)
    	{
    		throw new InvalidMessageException("Illegal value for seed count: " + e.getMessage());
    	}
    	catch (IllegalArgumentException e)
    	{
    		throw new InvalidMessageException("Illegal value for seed count: " + e.getMessage());
    	}

    	// 3rd argument: who's turn?
		moveTurn.end = false;
    	if (msgParts[3].equals("YOU\n"))
    		moveTurn.again = true;
    	else if (msgParts[3].equals("OPP\n"))
    		moveTurn.again = false;
    	else if (msgParts[3].equals("END\n"))
    	{
    		moveTurn.end = true;
    		moveTurn.again = false;
    	}
    	else
    		throw new InvalidMessageException("Illegal value for turn parameter: " + msgParts[3]);

    	return moveTurn;
    }
}


package MKAgent;
/**
 * Types of messages the game engine can send to the agent.
 */
public enum MsgType
{
	/**
	 * message announcing the start of the game ("new_match" message)
	 */
	START,
	/**
	 * message describing a move or a swap ("state_change" message)
	 */
	STATE,
	/**
	 * message informing about the end of the game ("game_over" message)
	 */
	END
}

package MKAgent;

import MKAgent.game.Board;
import MKAgent.game.Side;
import MKAgent.protocol.InvalidMessageException;
import MKAgent.protocol.MsgType;
import MKAgent.protocol.Protocol;
import MKAgent.treesearch.AlphaBetaTree;
import MKAgent.treesearch.TreeSearch;

import java.io.*;

/**
 * The main application class. It also provides methods for communication
 * with the game engine.
 */
public class Main {
    /**
     * Input from the game engine.
     */
    private static Reader input = new BufferedReader(new InputStreamReader(System.in));

    /**
     * Sends a message to the game engine.
     *
     * @param msg The message.
     */
    private static void sendMsg(String msg) {
        System.out.print(msg);
        System.out.flush();
    }

    /**
     * Receives a message from the game engine. Messages are terminated by
     * a '\n' character.
     *
     * @return The message.
     * @throws IOException if there has been an I/O error.
     */
    private static String recvMsg() throws IOException {
        StringBuilder message = new StringBuilder();
        int newCharacter;

        do {
            newCharacter = input.read();
            if (newCharacter == -1)
                throw new EOFException("Input ended unexpectedly.");
            message.append((char) newCharacter);
        } while ((char) newCharacter != '\n');

        return message.toString();
    }

    /**
     * The main method, invoked when the program is started.
     *
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        try {
            String msg;
            Board board = new Board(7, 7);
            Side ourSide = Side.SOUTH;
            boolean canSwap = false;
            TreeSearch treeSearch = new AlphaBetaTree();
            while (true) {
                System.err.println();
                msg = recvMsg();
                System.err.print("Received: " + msg);
                try {
                    MsgType msgType = Protocol.getMessageType(msg);
                    switch (msgType) {
                        case START:
                            System.err.println("A start.");
                            boolean first = Protocol.interpretStartMsg(msg);
                            if (first) {
                                int bestMoveIndex = treeSearch.getBestMove(board, ourSide);
                                sendMsg(Protocol.createMoveMsg(bestMoveIndex));
                            } else {
                                ourSide = Side.NORTH;
                                canSwap = true;
                            }
                            System.err.println("Starting player? " + first);
                            break;
                        case STATE:
                            System.err.println("A state.");
                            Protocol.MoveTurn move_turn = Protocol.interpretStateMsg(msg, board);
                            System.err.println("This was the move: " + move_turn.move);
                            System.err.println("Is the game over? " + move_turn.end);
                            System.err.println("Is it our turn again? " + move_turn.again);
                            if (move_turn.again) {
                                // out turn again
                                int move = treeSearch.getBestMove(board, ourSide);
                                sendMsg(Protocol.createMoveMsg(move));
                                // our turn
                                if (canSwap) {
                                    // always swap
                                    sendMsg(Protocol.createSwapMsg());
                                    ourSide = ourSide.opposite();
                                    canSwap = false;
                                    break;
                                }
                                int bestMoveIndex = treeSearch.getBestMove(board, ourSide);
                                sendMsg(Protocol.createMoveMsg(bestMoveIndex));
                            }
                            System.err.print("The board:\n" + board);
                            break;
                        case END:
                            System.err.println("An end. Bye bye!");
                            return;
                    }

                } catch (InvalidMessageException e) {
                    System.err.println(e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("This shouldn't happen: " + e.getMessage());
        }
    }
}

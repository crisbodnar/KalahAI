package MKAgent;

import MKAgent.game.Kalah;
import MKAgent.game.Side;
import MKAgent.protocol.InvalidMessageException;
import MKAgent.protocol.MsgType;
import MKAgent.protocol.Protocol;
import MKAgent.treesearch.TreeSearch;
import MKAgent.treesearch.mcts.MonteCarlo;

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
    private static String recMsg() throws IOException {
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
            Kalah state = new Kalah(7, 7);
            TreeSearch treeSearch = new MonteCarlo();
            while (true) {
                msg = recMsg();
                System.err.print("Received: " + msg);
                try {
                    MsgType msgType = Protocol.getMessageType(msg);
                    switch (msgType) {
                        case START:
                            System.err.println("A start.");
                            boolean first = Protocol.interpretStartMsg(msg);
                            if (first) {
                                int bestMoveIndex = treeSearch.getBestMove(state);
                                sendMsg(Protocol.createMoveMsg(bestMoveIndex));
                            } else {
                                state.setOurSide(Side.NORTH);
                            }
                            System.err.println("Starting player? " + first);
                            break;
                        case STATE:
                            System.err.println("A state.");
                            Protocol.MoveTurn move_turn = Protocol.interpretStateMsg(msg);
                            System.err.println("This was the move: " + move_turn.move);
                            System.err.println("Is the game over? " + move_turn.end);
                            System.err.println("Is it our turn again? " + move_turn.again);
                            state.makeMove(move_turn.move);
                            treeSearch.performMove(move_turn.move);
                            if (move_turn.again) {
                                int bestMoveIndex = treeSearch.getBestMove(state);
                                if (bestMoveIndex == 0) {
                                    sendMsg(Protocol.createSwapMsg());
                                } else {
                                    sendMsg(Protocol.createMoveMsg(bestMoveIndex));
                                }
                            }
                            System.err.print("The board:\n" + state.getBoard());
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

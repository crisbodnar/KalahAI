package MKAgent;

import MKAgent.game.Board;
import MKAgent.protocol.MsgType;
import MKAgent.protocol.Protocol;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;

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
    public static void sendMsg(String msg) {
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
    public static String recvMsg() throws IOException {
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
            String s;
            Board board = new Board(7, 7);
            while (true) {
                System.err.println();
                s = recvMsg();
                System.err.print("Received: " + s);
                try {
                    MsgType mt = Protocol.getMessageType(s);
                    switch (mt) {
                        case START:
                            System.err.println("A start.");
                            boolean first = Protocol.interpretStartMsg(s);
                            System.err.println("Starting player? " + first);
                            break;
                        case STATE:
                            System.err.println("A state.");
                            Protocol.MoveTurn r = Protocol.interpretStateMsg(s, board);
                            System.err.println("This was the move: " + r.move);
                            System.err.println("Is the game over? " + r.end);
                            if (!r.end) System.err.println("Is it our turn again? " + r.again);
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

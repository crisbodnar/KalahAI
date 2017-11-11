Protocol
========


(The notation used in the following for describing the syntax of the rules is a
 Backus–Naur Form (BNF) with the following extensions:
 - parentheses () represent a grouping of the enclosed symbols
 - an asterisk * denotes arbitrary repetition (including 0) of the preceding
   element
 - an asterisk followed by a number n (e.g. *17) means n-fold repetition of the
   preceding element
 For example the expression:
    "a" ("b" | "c")*2 "d"*
 would include the strings "abc", "accdddd", "acbd", ...)


All messages are of the following form:

   <MSG_NAME>  (";" <ARGUMENT>)*  <NL>

where <NL> is the character 0x0A ("\n" in C notation). To simplify things, no
spaces are allowed. The protocol is case sensitive.


Messages from the game engine to the agent (game engine -> agent):

- new_match:

  This message tells an agent that the game starts.

     "START" ";" <POSITION> <NL>

  <POSITION> ::= "North" | "South"
     The side of the board the agent is assigned.

  Note: South always starts the game, so "South" means the agent is the first
        player, while "North" means the agent is the second player.

- state_change:

  This message informs the agent about the last move and the resulting board
  configuration.

     "CHANGE" ";" <MOVESWAP> ";" <STATE> ";" <TURN> <NL>

  <MOVESWAP> ::= <MOVE> | "SWAP"
  <MOVE> ::= "1" | "2" | ... | n
     where n is the number of holes per side.

     A move i (with 1 <= i <= n) means a move which starts by picking the seeds
     of hole number i (of the player that made the move). Holes are numbered
     per side, starting with 1 on "the left" (i.e. furthest away from the
     player's store, the numbers increase in playing direction).

     A "SWAP" means that the opponent chose to swap the places after the
     starting player's first move, according to the pie rule (or swap rule).
     
  <STATE> ::= (<NAT> ",")*n <NAT> "," (<NAT> ",")*n <NAT>
  <NAT> ::= "0" | "1" | "2" | ... | k
     where n is the number of holes per side and k the number of seeds in the
     game.

     This gives the state of the board after the move. Informally, <STATE> is a
     comma separated list of 2*(n+1) natural numbers. Each number gives the
     number of seeds in a particular hole or store:
     - the first n numbers describe the holes of the North side, starting with
       hole 1,
     - the next number describes the Northern store,
     - the next n numbers describe the holes of the South side, starting with
       hole 1,
     - the next (and last) number describes the Southern store.

  <TURN> ::= "YOU" | "OPP" | "END"
     Says who's turn it is next ("YOU" meaning the agent, "OPP" the opponent
     once more). If turn is "END" the move terminated the match (regularly). In
     that case there will still be a "game_over" message after this one.

  Note: an agent always gets informed about all moves of the game, both their
        own and their opponent's ones (and hence also about the resulting board
        configurations), with the exception that an agent doesn't get informed
        if they chose to swap, only the opponent does.

- game_over:

  This message tells an agent that the game is finished.

     "END" <NL>

  Note: no more messages are sent by the game engine to the agent after this
        message. The agent should terminate after this message. If the match
        finished regularly (by a move after which no seeds are left in the
        moving agent's holes) there will have been a "state_change" message
        with <TURN> = "END" before. However, upon abnormal termination
        (timeout, illegal move by one of the agents, ...), an agent won't
        receive a "state_change,END" message but they will always receive a
        "game_over" message.



Messages from the agent to the game engine (agent -> game engine):

- move:

  This message informs the game engine about the agent's move.

     "MOVE" ";" <MOVE> <NL>

  where <MOVE> is the same as in the state_change message, i.e.:

  <MOVE> ::= "1" | "2" | ... | n
     where n is the number of holes per side.

     A move i (with 1 <= i <= n) means a move which starts by picking the seeds
     of (the agent's) hole number i. Holes are numbered per side, starting with
     1 on "the left" (i.e. furthest away from the player's store, the numbers
     increase in playing direction).

- swap:

  This message informs the game engine that the agent wants to swap sides,
  according to the pie rule (or swap rule).

     "SWAP" <NL>





See the state diagram for which messages can occur when.

The informal message names given in the state diagram mean the following:

(game engine -> agent):
- "new match, 1st":  "new_match" message with <POSITION> = "South"
- "new match, 2nd":  "new_match" message with <POSITION> = "North"
- "state, you":  "state_change" message with <TURN> = "YOU"
- "state, opponent":  "state_change" message with <TURN> = "OPP"
- "state, end":  "state_change" message with <TURN> = "END"
- "game over":  "game_over" message
- "state, you [swap]":  "state_change" message with <TURN> = "YOU" and
                        <MOVESWAP> = "SWAP"

(agent -> game engine):
- "move":  "move" message
- "swap":  "swap" message

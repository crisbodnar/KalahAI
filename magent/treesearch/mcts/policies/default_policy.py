import numpy as np

from magent.mancala import MancalaEnv
from magent.move import Move
from magent.treesearch import evaluation
from magent.treesearch.mcts.graph.node import AlphaNode, Node
from models.client import A3Client


class DefaultPolicy(object):
    """ DefaultPolicy plays out the domain from a given non-terminal state to produce a value estimate (simulation). """

    def simulate(self, root: Node) -> MancalaEnv:
        """ simulate run the game from given node and saves the reward for taking actions. """
        raise NotImplementedError("Simulate method is not implemented")


class MonteCarloDefaultPolicy(DefaultPolicy):
    """MonteCarloDefaultPolicy plays the domain randomly from a given non-terminal state."""

    @staticmethod
    def _make_temp_child(parent: Node, move: Move) -> MancalaEnv:
        child_state = MancalaEnv.clone(parent.state)
        child_state.perform_move(move)
        return child_state

    def simulate(self, root: Node) -> MancalaEnv:
        node = Node.clone(root)
        while not node.is_terminal():
            legal_moves = node.state.get_legal_moves()
            moves = [-1e80 for _ in range(node.state.board.holes + 1)]
            for move in legal_moves:
                moves[move.index] = evaluation.get_score(state=self._make_temp_child(node, move),
                                                         parent_side=node.state.side_to_move)

            moves_dist = np.asarray(moves, dtype=np.float64).flatten()
            exp = np.exp(moves_dist - np.max(moves_dist))
            dist = exp / np.sum(exp)

            move_to_make = int(np.random.choice(range(len(moves)), p=dist))
            node.state.perform_move(Move(side=node.state.side_to_move, index=move_to_make))

        return node.state


class AlphaGoDefaultPolicy(DefaultPolicy):
    """plays the domain based on prior probability provided by a neuron network. Starting at non-terminal state."""

    def __init__(self, network: A3Client):
        super(AlphaGoDefaultPolicy, self).__init__()
        self.network = network

    def simulate(self, root: AlphaNode) -> float:
        """
            runs a simulation from the root to the end of the game
            :param root: the starting node for the simulation
            :return: the rollout policy; reward for taking this path combining value network with game's winner
        """
        node: AlphaNode = AlphaNode.clone(root)
        while not node.is_terminal():
            move_index, _ = self.network.sample_state(node.state)
            move = Move(node.state.side_to_move, move_index + 1)
            node.state.perform_move(move)

        return node.state

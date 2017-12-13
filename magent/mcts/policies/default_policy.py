import logging
from random import choice

from magent.mancala import MancalaEnv
from magent.mcts.graph.node import AlphaNode, Node
from magent.move import Move
from models.client import A3Client


class DefaultPolicy(object):
    """ DefaultPolicy plays out the domain from a given non-terminal state to produce a value estimate (simulation). """

    def simulate(self, root: Node) -> MancalaEnv:
        """ simulate run the game from given node and saves the reward for taking actions. """
        raise NotImplementedError("Simulate method is not implemented")


class MonteCarloDefaultPolicy(DefaultPolicy):
    """MonteCarloDefaultPolicy plays the domain randomly from a given non-terminal state."""

    def simulate(self, root: Node) -> MancalaEnv:
        node = Node.clone(root)
        while not node.is_terminal():
            legal_move = choice(node.state.get_legal_moves())
            node.state.perform_move(legal_move)

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

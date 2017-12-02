from random import choice

from magent.mcts.graph.node import Node


# DefaultPolicy plays out the domain from a given non-terminal state to produce a value estimate (simulation).
class DefaultPolicy(object):
    # simulate run the game from given node and saves the reward for taking actions
    @staticmethod
    def simulate(node: Node):
        raise NotImplementedError("Select method is not implemented")


# MonteCarloDefaultPolicy plays the domain randomly from a given non-terminal state
class MonteCarloDefaultPolicy(DefaultPolicy):
    @staticmethod
    def simulate(node: Node):
        while not node.is_terminal():
            legal_move = choice(node.state.get_legal_moves())
            node.update(node.state.perform_move(legal_move))

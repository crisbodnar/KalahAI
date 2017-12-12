import logging
from random import choice

from magent.mancala import MancalaEnv
from magent.mcts.graph.node import AlphaNode, Node
from magent.move import Move


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

    def __init__(self, network):
        super(AlphaGoDefaultPolicy, self).__init__()
        self.network = network

    def simulate(self, root: AlphaNode, lmbd=1) -> float:
        """
            runs a simulation from the root to the end of the game
            :param root: the starting node for the simulation
            :param lmbd: a parameter to control the weight of the value network
            :return: the rollout policy; reward for taking this path combining value network with game's winner
        """
        node: AlphaNode = AlphaNode.clone(root)
        value = 0
        while not node.is_terminal():
            move_index, value = self.network.sample_state(node.state)
            move = Move(node.state.side_to_move, move_index + 1)
            node.state.perform_move(move)

        side_final_reward = node.state.compute_end_game_reward(root.state.side_to_move)
        reward = (1 - lmbd) * value + (lmbd * side_final_reward)
        logging.debug("Reward: %f; side final reward: %f; Value: %f" % (reward, side_final_reward, value))
        return reward  # (move reward + value network reward)

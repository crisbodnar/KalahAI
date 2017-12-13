from magent.mancala import MancalaEnv
from magent.mcts.graph.node import Node
from models.client import A3Client


class RollOutPolicy(object):
    """RollOutPolicy has the strategy for propagating rewards up the tree"""

    def backpropagate(self, node: Node, final_state: MancalaEnv):
        """Backpropgate starting at node the reward to parents"""
        raise NotImplementedError("backpropagate method is not implemented")


class MonteCarloRollOutPolicy(RollOutPolicy):
    def backpropagate(self, root: Node, final_state: MancalaEnv):
        """
        backpropgate pushes the reward (pay/visits) to the parents node up to the root
        :param root: starting node to backpropgate from
        :param final_state: the state of final node (holds final reward from the simulation)
        """
        node = root
        # propagate node reward to parents'
        while node is not None:
            side = node.parent.state.side_to_move if node.parent is not None else node.state.side_to_move  # root node
            node.update(final_state.compute_end_game_reward(side))
            node = node.parent


class AlphaGoRollOutPolicy(RollOutPolicy):
    def __init__(self, network: A3Client):
        """
        :param network: value network
        """
        super(AlphaGoRollOutPolicy, self).__init__()
        self.network = network

    def backpropagate(self, root: Node, final_state: MancalaEnv, lmbd=1):
        """backpropgate pushes the reward (pay/visits) to the parents node starting from the root down
            :param root: starting node to backpropgate from
            :param final_state: the state of final node (holds final reward from the simulation)
            :param lmbd: a parameter to control the weight of the value network
        """
        path_stack = []
        node = root
        # propagate node reward to parents'
        while node is not None:
            path_stack.append(node)
            node = node.parent
        # Update from root downward so the exploration bonus calculation is correct
        while len(path_stack) > 0:
            node = path_stack.pop()
            side = node.parent.state.side_to_move if node.parent is not None else node.state.side_to_move  # root node
            game_reward = final_state.compute_end_game_reward(side)
            # _, value = self.network.evaluate_state(final_state)
            # game_reward = (1 - lmbd) * value + (lmbd * side_final_reward) # value from network + value from actionNet
            node.update(game_reward)

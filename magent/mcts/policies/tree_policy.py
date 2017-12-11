from random import choice

from magent.mancala import MancalaEnv
from magent.mcts.graph.node import AlphaNode, Node
from magent.mcts.graph.node_utils import select_best_child, select_child_with_maximum_action_value
from magent.move import Move


class TreePolicy(object):
    """TreePolicy selects and expands from the nodes already contained within the search tree."""

    @staticmethod
    def select(node: Node) -> Node:
        raise NotImplementedError("Select method is not implemented")

    @staticmethod
    def expand(node: Node) -> Node:
        raise NotImplementedError("Expand method is not implemented")


class MonteCarloTreePolicy(TreePolicy):
    @staticmethod
    def select(node: Node) -> Node:
        while not node.is_terminal():
            # expand while we have nodes to expand
            if not node.is_fully_expanded():
                return MonteCarloTreePolicy.expand(node)
            # select child and explore it
            else:
                node = select_best_child(node)
        return node

    @staticmethod
    def expand(node: Node) -> Node:
        child_expansion_move = choice(tuple(node.unexplored_moves))
        child_state = MancalaEnv.clone(node.state)
        move_reward = child_state.perform_move(child_expansion_move)
        child_node = Node(state=child_state, move=child_expansion_move, parent=node)
        child_node.update(move_reward)
        node.put_child(child_node)
        # go down the tree
        return child_node


class AlphaGoTreePolicy(TreePolicy):
    def __init__(self, network):
        super(AlphaGoTreePolicy, self).__init__()
        self.network = network

    def select(self, node: AlphaNode) -> AlphaNode:
        while not node.is_terminal():
            # expand while we have nodes to expand
            if not node.is_fully_expanded():
                return self.expand(node)
            # select child and explore it
            else:
                # Select action among children that gives maximum action value, Q plus bonus u(P).
                node = select_child_with_maximum_action_value(node)
        return node

    def expand(self, node: AlphaNode):
        # Tactical workaround the pie move
        if Move(node.state.side_to_move, 0) in node.unexplored_moves:
            node.unexplored_moves.remove(Move(node.state.side_to_move, 0))

        dist, value = self.network.evaluate_state(node.state)
        for index, prior in enumerate(dist):
            if prior != 0.0:
                child_state = MancalaEnv.clone(node.state)
                expansion_move = Move(child_state.side_to_move, index + 1)
                child_state.perform_move(expansion_move)
                child_node = AlphaNode(state=child_state, prior=prior, move=expansion_move, parent=node)
                node.put_child(child_node)
        # go down the tree
        return select_child_with_maximum_action_value(node)

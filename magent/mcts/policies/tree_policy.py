from random import choice

from magent.mancala import MancalaEnv
from magent.mcts import evaluation
from magent.mcts.graph import node_utils
from magent.mcts.graph.node import AlphaNode, Node
from magent.move import Move
from models.client import A3Client


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
                node = node_utils.rave_selection(node)
        return node

    @staticmethod
    def expand(parent: Node) -> Node:
        child_expansion_move = choice(tuple(parent.unexplored_moves))
        child_state = MancalaEnv.clone(parent.state)
        child_state.perform_move(child_expansion_move)
        child_node = Node(state=child_state, move=child_expansion_move, parent=parent)
        child_node.value = evaluation.evaluate_node(state=child_state, parent_side=parent.state.side_to_move)
        parent.put_child(child_node)
        # go down the tree
        return child_node


class AlphaGoTreePolicy(TreePolicy):
    def __init__(self, network: A3Client):
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
                node = node_utils.select_child_with_maximum_action_value(node)
        return node

    def expand(self, node: AlphaNode):
        # Tactical workaround the pie move
        if Move(node.state.side_to_move, 0) in node.unexplored_moves:
            node.unexplored_moves.remove(Move(node.state.side_to_move, 0))

        dist, value = self.network.evaluate_state(node.state)
        for index, prior in enumerate(dist):
            expansion_move = Move(node.state.side_to_move, index + 1)
            if node.state.is_legal(expansion_move):
                child_state = MancalaEnv.clone(node.state)
                child_state.perform_move(expansion_move)
                child_node = AlphaNode(state=child_state, prior=prior, move=expansion_move, parent=node)
                node.put_child(child_node)
        # go down the tree
        return node_utils.select_child_with_maximum_action_value(node)

from random import choice

import numpy as np

from magent.mancala import MancalaEnv
from magent.move import Move
from magent.treesearch import evaluation
from magent.treesearch.mcts.graph import node_utils
from magent.treesearch.mcts.graph.node import Node, AlphaNode
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
        parent.put_child(child_node)
        MonteCarloTreePolicy._rave_expand(child_node)
        # go down the tree
        return child_node

    @staticmethod
    def _rave_expand(parent: Node):
        moves = [-1e80 for _ in range(parent.state.board.holes + 1)]
        for unexplored_move in parent.unexplored_moves.copy():
            child_state = MancalaEnv.clone(parent.state)
            child_state.perform_move(unexplored_move)
            moves[unexplored_move.index] = evaluation.get_score(state=child_state,
                                                                parent_side=parent.state.side_to_move)

        moves_dist = np.asarray(moves, dtype=np.float64).flatten()
        exp = np.exp(moves_dist - np.max(moves_dist))
        dist = exp / np.sum(exp)
        parent.value = max(dist)


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

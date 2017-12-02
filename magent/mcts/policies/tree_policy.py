from random import choice

from magent.mancala import MancalaEnv
from magent.mcts.graph.node import Node
from magent.mcts.graph.node_utils import select_best_child


# TreePolicy selects and expands from the nodes already contained within the search tree.
class TreePolicy(object):
    @staticmethod
    def select(node: Node) -> Node:
        raise NotImplementedError("Select method is not implemented")


class MonteCarloTreePolicy(TreePolicy):
    @staticmethod
    def select(node: Node) -> Node:
        while not node.is_terminal():
            # expand while we have nodes to expand
            if not node.is_fully_expanded():
                return MonteCarloTreePolicy._expand(node)
            # select child and explore it
            else:
                node = select_best_child(node)
        return node

    @staticmethod
    def _expand(node: Node) -> Node:
        child_expansion_move = choice(tuple(node.unexplored_moves))
        child_state = MancalaEnv.clone(node.state)
        move_reward = child_state.perform_move(child_expansion_move)
        child_node = Node(state=child_state, move=child_expansion_move, parent=node)
        child_node.update(move_reward)
        node.put_child(child_node)
        # go down the tree
        return child_node

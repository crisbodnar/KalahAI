from math import log, sqrt

from magent.treesearch.mcts.graph.node import Node, AlphaNode


def select_best_child(node: Node) -> Node:
    """select_best_child returns the child that maximise upper confidence interval (UCT applied to trees)."""
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]

    return max(node.children, key=lambda child: _uct_reward(node, child))


def select_max_child(node: Node) -> Node:
    """select_max_child returns the child with highest average reward."""
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    if len(node.children) == 0:
        raise ValueError('Selecting max child from unexpanded node')
    elif len(node.children) == 1:
        return node.children[0]
    return max(node.children, key=lambda child: child.reward / child.visits)


def select_robust_child(node: Node) -> Node:
    """select_robust_child returns the child that is most visited."""
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]
    return max(node.children, key=lambda child: child.visits)


def select_secure_child(node: Node) -> Node:
    """select_secure_child returns child which maximises a lower confidence interval (LCT applied to trees)."""
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]

    return max(node.children, key=lambda child: _lower_confidence_interval(node, child))


def select_child_with_maximum_action_value(node: AlphaNode) -> AlphaNode:
    return max(node.children, key=lambda child: child.calculate_action_value())


def _uct_reward(root: Node, child: Node, exploration_constant: float = 1 / sqrt(2)) -> float:
    return (child.reward / child.visits) + (exploration_constant * sqrt(2 * log(root.visits) / child.visits))


def _lower_confidence_interval(root: Node, child: Node, exploration_constant: float = sqrt(2)) -> float:
    return (child.reward / child.visits) - (exploration_constant * sqrt(2 * log(root.visits) / child.visits))


def rave_selection(node: Node) -> Node:
    """returns the child that maximise the heuristic value."""
    if node.is_terminal() and node.is_fully_expanded():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]

    return max(node.children, key=lambda child: _uct_rave_reward(node, child))


def _uct_rave_reward(root: Node, child: Node, exploration_constant: float = 0.5) -> float:
    return _rave_reward(child) + (exploration_constant * sqrt(log(root.visits) / child.visits))


def _rave_reward(node: Node, alpha: float = 0.5) -> float:
    return (1 - alpha) * (node.reward / node.visits) + alpha * node.value

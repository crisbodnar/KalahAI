from math import sqrt, log
from magent.mcts.graph.node import Node


# backpropgate pushes the reward (pay/visits) to the parents node up to the root
def backpropagate(node: Node):
    node_reward = node.reward
    parent = node.parent
    # propagate node reward to parents'
    while parent is not None:
        parent.update(node_reward)
        parent = parent.parent


# select_best_child returns the child that maximise upper confidence interval (UCT applied to trees)
def select_best_child(node: Node) -> Node:
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]

    max_child, max_child_reward = (node.children[0], _uct_reward(node, node.children[0]))
    for child in node.children[1:]:
        child_reward = _uct_reward(node, child)
        if child_reward > max_child_reward:
            max_child = child
            max_child_reward = child_reward
    return max_child


# select_max_child returns the child with highest average reward
def select_max_child(node: Node) -> Node:
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    if len(node.children) == 0:
        raise ValueError('Selecting max child from unexpanded node')
    elif len(node.children) == 1:
        return node.children[0]
    return sorted(node.children, key=lambda child: child.reward / child.visits)[-1]


# select_robust_child returns the child that is most visited
def select_robust_child(node: Node) -> Node:
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]
    return sorted(node.children, key=lambda child: child.visits)[-1]


# select_secure_child returns child which maximises a lower confidence interval (LCT applied to trees)
def select_secure_child(node: Node) -> Node:
    if node.is_terminal():
        raise ValueError('Terminal node; there are no children to select from.')
    elif len(node.children) == 1:
        return node.children[0]

    max_child, max_lower_confidence_interval = (node.children[0], _lower_confidence_interval(node, node.children[0]))
    for child in node.children[1:]:
        _, child_reward = _lower_confidence_interval(node, child)
        if child_reward > max_lower_confidence_interval:
            max_child = child
            max_lower_confidence_interval = child_reward
    return max_child


def _uct_reward(root: Node, child: Node, exploration_constant: float = 1 / sqrt(2)) -> float:
    return (child.reward / child.visits) + (exploration_constant * sqrt(2 * log(root.visits) / child.visits))


def _lower_confidence_interval(root: Node, child: Node, exploration_constant: float = 1 / sqrt(2)) -> float:
    return (child.reward / child.visits) - (exploration_constant * sqrt(2 * log(root.visits) / child.visits))

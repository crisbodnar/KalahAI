from magent.mancala import MancalaEnv
from magent.move import Move
from copy import deepcopy


class Node(object):
    def __init__(self, state: MancalaEnv, move: Move = None, parent=None):
        self.visits = 0
        self.reward = 0
        self.state = state
        self.children = []
        self.parent = parent
        self.move = move
        self.unexplored_moves = set(state.get_legal_moves())

    @staticmethod
    def clone(other_node):
        return deepcopy(other_node)

    def put_child(self, child):
        self.children.append(child)
        self.unexplored_moves.remove(child.move)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def unvisited_children(self):
        nodes = []
        for child in self.children:
            if child.visits < 1:
                nodes.append(child)
        return nodes

    # is_fully_expanded returns true if there are no more moves to explore
    def is_fully_expanded(self) -> bool:
        return len(self.unexplored_moves) == 0

    # is_terminal returns true if the node is leaf node
    def is_terminal(self) -> bool:
        return len(self.state.get_legal_moves()) == 0

    # backpropgate pushes the reward (pay/visits) to the parents node up to the root
    def backpropagate(self, reward):
        parent = self.parent
        # propagate node reward to parents'
        while parent is not None:
            parent.update(reward)
            parent = parent.parent

    def __str__(self):
        return "Node; Move %s, number of children: %d; visits: %d; reward: %f" % (
            self.move, len(self.children), self.visits, self.reward)

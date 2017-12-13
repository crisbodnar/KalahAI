from copy import deepcopy

from numpy.ma import sqrt

from magent.mancala import MancalaEnv
from magent.move import Move


class Node(object):
    def __init__(self, state: MancalaEnv, move: Move = None, parent=None):
        self.visits = 0
        self.reward = 0
        self.state = state
        self.children = []
        self.parent = parent
        self.move = move
        self.value = -1
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
        for child in self.children:
            self.value = max(self.value, child.value)

    def _make_temp_child(self, move: Move) -> MancalaEnv:
        child_state = MancalaEnv.clone(self.state)
        child_state.perform_move(move)
        return child_state

    # def rave_update(self):

    def is_fully_expanded(self) -> bool:
        """ is_fully_expanded returns true if there are no more moves to explore. """
        return len(self.unexplored_moves) == 0

    def is_terminal(self) -> bool:
        """is_terminal returns true if the node is leaf node"""
        return len(self.state.get_legal_moves()) == 0

    def __str__(self):
        return "Node; Move %s, number of children: %d; visits: %d; reward: %f; value: %f" % (
            self.move, len(self.children), self.visits, self.reward, self.value)


class AlphaNode(Node):
    def __init__(self, state: MancalaEnv, prior, move: Move = None, parent=None):
        super(AlphaNode, self).__init__(state, move, parent)
        # u(s,a) - probational to prior probability but decays with repeated visits to encourage exploration.
        self.exploration_bonus = prior / (1 + self.visits)  # u(s,a) exploration bonus
        self.prior = prior  # P(s,a) prior probability

    def update(self, reward: float, c_puct: int = 5):
        """
            :param reward: leaf reward
            :param c_puct: a constant determining the level of exploration (PUCT algorithm)
        """
        self.visits += 1
        self.reward += ((reward - self.reward) / self.visits)
        if self.parent is not None:
            self.exploration_bonus = c_puct * self.prior * sqrt(self.parent.visits) / (1 + self.visits)

    def calculate_action_value(self) -> float:
        return self.reward + self.exploration_bonus

    def __str__(self):
        return "Node; Move %s, number of children: %d; visits: %d; reward: %f; exploration bonus: %f; prior: %f" % (
            self.move, len(self.children), self.visits, self.reward, self.exploration_bonus, self.prior)

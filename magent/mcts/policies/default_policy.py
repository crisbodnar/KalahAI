from random import choice
from magent.mcts.graph.node import Node, AlphaNode

# DefaultPolicy plays out the domain from a given non-terminal state to produce a value estimate (simulation).
from magent.move import Move


class DefaultPolicy(object):
    # simulate run the game from given node and saves the reward for taking actions
    @staticmethod
    def simulate(root: Node) -> float:
        raise NotImplementedError("Select method is not implemented")


# MonteCarloDefaultPolicy plays the domain randomly from a given non-terminal state
class MonteCarloDefaultPolicy(DefaultPolicy):
    @staticmethod
    def simulate(root: Node) -> float:
        node = Node.clone(root)
        while not node.is_terminal():
            legal_move = choice(node.state.get_legal_moves())
            node.update(node.state.perform_move(legal_move))
        return node.reward


class AlphaGoDefaultPolicy(DefaultPolicy):
    def __init__(self, network):
        super(AlphaGoDefaultPolicy, self).__init__()
        self.neuro_net = network

    def simulate(self, root: AlphaNode, lmbd=0.5) -> float:
        """
            runs a simulation from the root to the end of the game
            :param root: the starting node for the simulation
            :param lmbd: a parameter to control the weight of the value network
            :return: the rollout policy; reward for taking this path combining value network with game's winner
        """
        node: AlphaNode = Node.clone(root)
        value = 0
        while not node.is_terminal():
            best_move, _, value = self.neuro_net.get_best_move(node.state)
            legal_move = Move(node.state.side_to_move, best_move)
            node.update(node.state.perform_move(legal_move))

        reward = (1 - lmbd) * value + (lmbd * node.reward)

        return reward  # (move reward + value network reward)

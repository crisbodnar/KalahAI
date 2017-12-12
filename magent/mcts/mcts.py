import datetime
import logging

import magent.mcts.graph.node_utils as node_utils
from magent.mancala import MancalaEnv
from magent.mcts.graph.node import Node
from magent.mcts.policies.default_policy import AlphaGoDefaultPolicy, DefaultPolicy, MonteCarloDefaultPolicy
from magent.mcts.policies.tree_policy import AlphaGoTreePolicy, MonteCarloTreePolicy, TreePolicy
from magent.move import Move


class MCTS(object):
    def __init__(self, tree_policy: TreePolicy, default_policy: DefaultPolicy, time_sec: int):
        self.tree_policy: TreePolicy = tree_policy
        self.default_policy: DefaultPolicy = default_policy
        self.calculation_time: datetime.timedelta = datetime.timedelta(seconds=time_sec)

    def search(self, state: MancalaEnv) -> Move:
        # short circuit last move
        if len(state.get_legal_moves()) == 1:
            return state.get_legal_moves()[0]

        game_state_root = Node(state=MancalaEnv.clone(state))
        start_time = datetime.datetime.utcnow()
        games_played = 0
        while datetime.datetime.utcnow() - start_time < self.calculation_time:
            node = self.tree_policy.select(game_state_root)

            final_state = self.default_policy.simulate(node)
            node.backpropagate(final_state)
            # Debugging information
            games_played += 1
            logging.debug("%s; Game played %i" % (node, games_played))
        logging.debug("%s" % game_state_root)
        robust_child = node_utils.select_robust_child(game_state_root)
        logging.info("Choosing: %s" % robust_child)
        return robust_child.move


class MCTSFactory(object):
    """Factory class to load various MCTS configurations."""

    @staticmethod
    def standard_mcts() -> MCTS:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    time_sec=30)

    @staticmethod
    def test_mcts() -> MCTS:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    time_sec=10)

    @staticmethod
    def alpha_mcts(network_client) -> MCTS:
        return MCTS(tree_policy=AlphaGoTreePolicy(network_client),
                    default_policy=AlphaGoDefaultPolicy(network_client),
                    time_sec=30)

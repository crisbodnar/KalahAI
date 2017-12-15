from magent.treesearch.alphabeta.alphabeta import AlphaBeta
from magent.treesearch.mcts.mcts import MCTS
from magent.treesearch.mcts.policies.default_policy import MonteCarloDefaultPolicy, AlphaGoDefaultPolicy
from magent.treesearch.mcts.policies.rollout_policy import MonteCarloRollOutPolicy, AlphaGoRollOutPolicy
from magent.treesearch.mcts.policies.tree_policy import MonteCarloTreePolicy, AlphaGoTreePolicy
from magent.treesearch.treesearch import TreeSearch
from models.client import A3Client


class TreesFactory(object):
    """Factory class to load various MCTS configurations."""

    @staticmethod
    def standard_mcts() -> TreeSearch:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=20)

    @staticmethod
    def test_mcts() -> TreeSearch:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=10)

    @staticmethod
    def long_test_mcts(sec: int = 0) -> TreeSearch:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=sec)

    @staticmethod
    def alpha_mcts(network_client: A3Client) -> TreeSearch:
        return MCTS(tree_policy=AlphaGoTreePolicy(network_client),
                    default_policy=AlphaGoDefaultPolicy(network_client),
                    rollout_policy=AlphaGoRollOutPolicy(network_client),
                    time_sec=30)

    @staticmethod
    def alpha_beta() -> TreeSearch:
        return AlphaBeta(depth=5)

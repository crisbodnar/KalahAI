import datetime
import logging

import magent.mcts.graph.node_utils as node_utils
from magent.mancala import MancalaEnv
from magent.mcts.graph.node import Node
from magent.mcts.policies.default_policy import AlphaGoDefaultPolicy, DefaultPolicy, MonteCarloDefaultPolicy
from magent.mcts.policies.rollout_policy import AlphaGoRollOutPolicy, MonteCarloRollOutPolicy, RollOutPolicy
from magent.mcts.policies.tree_policy import AlphaGoTreePolicy, MonteCarloTreePolicy, TreePolicy
from magent.move import Move
from models.client import A3Client


class MCTS(object):
    def __init__(self, tree_policy: TreePolicy, default_policy: DefaultPolicy, rollout_policy: RollOutPolicy,
                 time_sec: int):
        self.tree_policy: TreePolicy = tree_policy
        self.default_policy: DefaultPolicy = default_policy
        self.rollout_policy = rollout_policy
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
            self.rollout_policy.backpropagate(node, final_state)
            # Debugging information
            games_played += 1
            logging.debug("%s; Game played %i" % (node, games_played))
        logging.debug("%s" % game_state_root)
        chosen_child = node_utils.select_robust_child(game_state_root)
        logging.info("Choosing: %s" % chosen_child)
        return chosen_child.move


class MCTSFactory(object):
    """Factory class to load various MCTS configurations."""

    @staticmethod
    def standard_mcts() -> MCTS:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=60)

    @staticmethod
    def test_mcts() -> MCTS:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=10)

    @staticmethod
    def long_test_mcts(sec: int = 0) -> MCTS:
        return MCTS(tree_policy=MonteCarloTreePolicy(),
                    default_policy=MonteCarloDefaultPolicy(),
                    rollout_policy=MonteCarloRollOutPolicy(),
                    time_sec=sec)

    @staticmethod
    def alpha_mcts(network_client: A3Client) -> MCTS:
        return MCTS(tree_policy=AlphaGoTreePolicy(network_client),
                    default_policy=AlphaGoDefaultPolicy(network_client),
                    rollout_policy=AlphaGoRollOutPolicy(network_client),
                    time_sec=30)

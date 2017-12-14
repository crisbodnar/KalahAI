import datetime
import logging

from magent.mancala import MancalaEnv
from magent.move import Move
from magent.treesearch.mcts.graph import node_utils
from magent.treesearch.mcts.graph.node import Node
from magent.treesearch.mcts.policies.default_policy import DefaultPolicy
from magent.treesearch.mcts.policies.rollout_policy import RollOutPolicy
from magent.treesearch.mcts.policies.tree_policy import TreePolicy
from magent.treesearch.treesearch import TreeSearch


class MCTS(TreeSearch):
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

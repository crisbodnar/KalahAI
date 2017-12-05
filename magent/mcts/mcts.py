import datetime
import logging

from magent.mancala import MancalaEnv
from magent.mcts.graph.node import Node
from magent.mcts.graph.node_utils import select_max_child, backpropagate
from magent.mcts.policies.policies import tree_factory, default_factory, DefaultPolicy, TreePolicy
from magent.move import Move


class MCTS(object):
    def __init__(self, tree_policy: TreePolicy, default_policy: DefaultPolicy, time_sec: int):
        self.tree_policy: TreePolicy = tree_policy
        self.default_policy: DefaultPolicy = default_policy
        self.calculation_time: datetime.timedelta = datetime.timedelta(seconds=time_sec)

    def search(self, state: MancalaEnv) -> Move:
        game_state_root = Node(state=MancalaEnv.clone(state))
        start_time = datetime.datetime.utcnow()
        games_played = 0
        while datetime.datetime.utcnow() - start_time < self.calculation_time:
            node = self.tree_policy.select(game_state_root)
            self.default_policy.simulate(node)
            backpropagate(node)
            # Debugging information
            games_played += 1
            logging.debug("%s; Game played %i" % (node, games_played))

        return select_max_child(game_state_root).move


mcts_presets = {
    'standard-mcts': MCTS(tree_policy=tree_factory('monte-carlo'),
                          default_policy=default_factory('monte-carlo'),
                          time_sec=60)
}


def mcts_factory(configuration_name: str) -> MCTS:
    return mcts_presets[configuration_name]

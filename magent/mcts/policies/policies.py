from magent.mcts.policies.tree_policy import MonteCarloTreePolicy, TreePolicy
from magent.mcts.policies.default_policy import MonteCarloDefaultPolicy, DefaultPolicy

tree_policies = {
    'monte-carlo': MonteCarloTreePolicy()
}

default_policies = {
    'monte-carlo': MonteCarloDefaultPolicy()
}


# tree_factory returns a tree policy associated with the given key
def tree_factory(name: str) -> TreePolicy:
    return tree_policies[name]


# default_factory returns a default policy associated with the given key
def default_factory(name: str) -> DefaultPolicy:
    return default_policies[name]

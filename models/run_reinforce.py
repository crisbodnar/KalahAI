from models.pg_reinforce import PolicyGradientAgent
from models.pg_trainer import PolicyGradientTrainer
from magent.mancala import MancalaEnv
import tensorflow as tf


def main():
    sess = tf.Session()
    south = PolicyGradientAgent(sess, name='south')
    north = PolicyGradientAgent(sess, name='north')
    env = MancalaEnv()

    pg_trainer = PolicyGradientTrainer(south, north, env)
    pg_trainer.train()
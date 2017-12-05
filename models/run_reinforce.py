from models.reinforce_agent import PolicyGradientAgent
from models.pg_trainer import PolicyGradientTrainer
from magent.mancala import MancalaEnv
import tensorflow as tf


def main(_):
    with tf.Session() as sess:
        agent = PolicyGradientAgent(sess, name='pg_agent', load_model=True)
        env = MancalaEnv()

        pg_trainer = PolicyGradientTrainer(agent, env)
        pg_trainer.train()


if __name__ == '__main__':
    tf.app.run()

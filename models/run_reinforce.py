from models.reinforce_agent import PolicyGradientAgent
from models.pg_trainer import PolicyGradientTrainer
from magent.mancala import MancalaEnv
import tensorflow as tf


def main(_):
    with tf.Session() as sess:
        south = PolicyGradientAgent(sess, name='south')
        north = PolicyGradientAgent(sess, name='north')

        try:
            south.restore_model_params('south')
            print('Successfully loaded checkpoint for {}.'.format(south.name))
        except:
            print('Failed to load checkpoint for {}.'.format(south.name))

        env = MancalaEnv()

        pg_trainer = PolicyGradientTrainer(south, north, env)
        pg_trainer.train()


if __name__ == '__main__':
    tf.app.run()
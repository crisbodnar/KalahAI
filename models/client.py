import logging

import tensorflow as tf

from magent.mancala import MancalaEnv
from magent.side import Side
from models.a3c.helpers import FastSaver
from models.a3c.model import ACNetwork
from models.tdlambda.model import Model
from models.tdlambda.agent import TDAgent
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A3Client(object):
    def __init__(self, sess):
        self.sess = sess
        self.network = ACNetwork(state_shape=[2, 8, 1], num_act=7)
        self._restore_model()

    def _restore_model(self):
        saver = FastSaver()
        try:
            logger.debug('Loading Model...')
            checkpoint_path = tf.train.get_checkpoint_state(checkpoint_dir="./saved_ckpt/a3c_random_0.9")
            saver.restore(sess=self.sess, save_path=checkpoint_path.model_checkpoint_path)
        except Exception as e:
            logger.error("Failed to restore models", str(e))

    def evaluate_state(self, env: MancalaEnv):
        flip_board = env.side_to_move == Side.NORTH
        state = env.board.get_board_image(flipped=flip_board)
        mask = env.get_action_mask_with_no_pie()
        dist, _, value = self.network.evaluate_move(state=state, mask=mask)

        return dist, float(value)

    def sample_state(self, env: MancalaEnv) -> (int, int):
        flip_board = env.side_to_move == Side.NORTH
        state = env.board.get_board_image(flipped=flip_board)
        mask = env.get_action_mask_with_no_pie()
        return self.network.sample(state=state, mask=mask)


class TDClient(object):
    def __init__(self, sess):
        self.sess = sess
        path = 'models/'
        if not os.path.exists(path):
            os.makedirs(path)

        self.network = Model(sess, path, path, 'checkpoints/', restore=True)
        self.td_agent = TDAgent(self.network)

    def sample_state(self, env: MancalaEnv) -> (int, int):
        return self.td_agent.get_action(env), 0

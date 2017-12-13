import time
import numpy as np
import tensorflow as tf

from magent.mancala import MancalaEnv
from magent.side import Side
from models.tdlambda.agent import TDAgent, RandomAgent


class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step,
                           30000, 0.96, staircase=True), name='lambda')

        # learning rate decay
        alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step,
                           40000, 0.96, staircase=True), name='alpha')

        tf.summary.scalar('lambda', lamda)
        tf.summary.scalar('alpha', alpha)

        w_init = tf.contrib.layers.xavier_initializer()

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, 20], name='x')
        self.V_next = tf.placeholder('float', [1, 1], name='V_next')

        # build network architecture
        flattened_imp = tf.contrib.layers.flatten(self.x)
        net_h1 = tf.layers.dense(inputs=flattened_imp, units=100, activation=tf.nn.sigmoid, kernel_initializer=w_init)
        net_h2 = tf.layers.dense(inputs=net_h1, units=80, activation=tf.nn.sigmoid, kernel_initializer=w_init)
        net_h3 = tf.layers.dense(inputs=net_h2, units=40, activation=tf.nn.sigmoid, kernel_initializer=w_init)
        net_h3 = tf.layers.dense(inputs=net_h3, units=30, activation=tf.nn.sigmoid, kernel_initializer=w_init)
        self.V = tf.layers.dense(inputs=net_h3, units=1, activation=tf.nn.sigmoid, kernel_initializer=w_init)


        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = tf.div(loss_sum, tf.maximum(game_step, 1.0))
            delta_avg_op = tf.div(delta_sum, tf.maximum(game_step, 1.0))
            accuracy_avg_op = tf.div(accuracy_sum, tf.maximum(game_step, 1.0))

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/accuracy_avg', accuracy_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.summary.scalar('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=5)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={ self.x: [x] })

    # def play(self):
    #     game = Game.new()
    #     game.play([TDAgent(Game.TOKENS[0], self), HumanAgent(Game.TOKENS[1])], draw=True)

    def test(self, episodes=100, draw=False):
        players = [TDAgent(self), RandomAgent()]
        winners = [0, 0]
        for episode in range(episodes):
            game = MancalaEnv()

            while not game.is_game_over():
                if game.side_to_move == Side.SOUTH:
                    move = players[Side.get_index(Side.SOUTH)].get_action(game)
                else:
                    move = players[Side.get_index(Side.NORTH)].get_action(game)
                game.perform_move(move)

            winner = game.get_winner()
            if winner is None:
                continue

            winner_side = Side.get_index(winner)
            winners[winner_side] += 1

            winners_total = sum(winners)
            print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                players[0].name, players[0].name,
                players[1].name, players[1].name,
                winners[0], winners[1], winners_total,
                (winners[0] / winners_total) * 100.0))

    def train(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path,
                                                               int(time.time()), self.sess.graph_def))

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(self, name='NORTH'), TDAgent(self, 'SOUTH')]

        validation_interval = 1000
        episodes = 50000

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            game = MancalaEnv()
            x = game.board.get_board_image_with_heuristics(game.side_to_move)

            game_step = 0
            while not game.is_game_over():
                if game.side_to_move == Side.SOUTH:
                    move = players[Side.get_index(Side.SOUTH)].get_action(game, explore=True,
                                                                          exploration_const=episode)
                    game.perform_move(move)
                else:
                    move = players[Side.get_index(Side.NORTH)].get_action(game, explore=True,
                                                                          exploration_const=episode)
                    game.perform_move(move)

                x_next = game.board.get_board_image_with_heuristics(game.side_to_move)
                V_next = self.get_output(x_next)
                self.sess.run(self.train_op, feed_dict={ self.x: [x], self.V_next: V_next })

                x = x_next
                game_step += 1

            winner = game.get_winner()
            payoff = 0.5
            if winner is not None:
                payoff = Side.get_index(winner)

            _, global_step, summaries, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.summaries_op,
                self.reset_op
            ], feed_dict={ self.x: [x], self.V_next: np.array([[payoff]], dtype='float') })
            summary_writer.add_summary(summaries, global_step=global_step)

            if winner is not None and episode != 0 and episode % 100 == 0:
                print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes,
                                                               players[Side.get_index(winner)].name, game_step))
                self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()

        # self.test(episodes=1000)

import threading
import multiprocessing
import tensorflow as tf

from time import sleep

from models.a3c.ac_network import ActorCriticNetwork
from models.a3c.worker import Worker
from magent.mancala import MancalaEnv
import os


max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
a_size = 7
load_model = False
model_path = './checkpoints/a3c'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=0.0002, decay=0.99, epsilon=0.1)
    master_network = ActorCriticNetwork(a_size, 'global', None)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(MancalaEnv(), i, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        def worker_work():
            worker.work(gamma, sess, coord, saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
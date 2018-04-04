import tensorflow as tf
import numpy as np
from dnn import DNN
from data import cifar10_input
from tqdm import tqdm
import time,os, sys
from libs.utils import suppress_stdout


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_integer('seed', 1546, 'seed[1]')
flags.DEFINE_boolean('cnn', True, 'seed[1]')

flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
flags.DEFINE_integer("epoch", 200, "batch size [128]")

FLAGS = flags.FLAGS
status_reporter = None #report status with ray
verbose = True


def main(_):
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.lower(), value))
    print("")

    rng = np.random.RandomState(FLAGS.seed)  # seed labels

    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')

    # select labeled data
    inds = rng.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)
    trainx = txs
    trainy = tys
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)
    print(trainx.shape)

    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate_pl = tf.placeholder(tf.float32, [], name='adam_learning_rate_pl')

    classifier = DNN()
    logits = classifier(inp, is_training=is_training_pl)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), lbl)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # control dependencies for batch norm ops
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    def linear_decay(decay_start, decay_end, epoch):
        return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start), 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(FLAGS.epoch),disable=not verbose):
            begin = time.time()

            # randomly permuted minibatches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            train_loss = train_acc = test_acc = 0

            lr = FLAGS.learning_rate * linear_decay(100, 200, epoch)

            for t in tqdm(range(nr_batches_train),disable=not verbose):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate_pl: lr}

                _, ls, acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                train_loss += ls
                train_acc += acc

            train_loss /= nr_batches_train
            train_acc /= nr_batches_train

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}

                test_acc += sess.run(accuracy, feed_dict=feed_dict)
            test_acc /= nr_batches_test

            tqdm.write("Epoch %03d | Time = %03ds | lr = %.4f | loss train = %.4f | train acc = %.4f | test acc = %.4f" %
                       (epoch, time.time() - begin, lr,train_loss, train_acc, test_acc))

            if status_reporter: # report status for ray tune
                status_reporter(timesteps_total=epoch, mean_accuracy=test_acc)


def train_with_dic(config=None, reporter=None):
    global status_reporter, verbose
    status_reporter = reporter
    verbose = False

    argv = []     # config dic => argv
    for key, value in config.items():
        argv.extend(['--' + key, str(value)])

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # warning tensorflow consol
    # with suppress_stdout(): # redirect ouptut to null
    tf.app.run(main=main, argv=[sys.argv[0]] + argv)


if __name__ == "__main__":
    tf.app.run()



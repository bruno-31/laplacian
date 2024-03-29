import tensorflow as tf
import numpy as np
from dnn import DNN, classifier
from data import cifar10_input
from tqdm import tqdm
import os, time, sys
from libs.utils import suppress_stdout

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_integer('seed', 1546, 'seed[1]')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
flags.DEFINE_float('ma_decay', 0.9999, 'exp moving average for inference [0.9999]')

FLAGS = flags.FLAGS
status_reporter = None #report status with ray


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.lower(), value))
    print("")
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    rng = np.random.RandomState(FLAGS.seed)  # seed labels

    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
    trainx_unl = trainx.copy()

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

    logits = classifier(inp, is_training=is_training_pl)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), lbl)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # control dependencies for batch norm ops

    dvars = [var for var in tf.trainable_variables() if 'classifier' in var.name]
    [print(var) for var in dvars]
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, var_list=dvars)

    ### ema ###
    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
    maintain_averages_op = ema.apply(dvars)

    with tf.control_dependencies([train_op]):
        train_dis_op = tf.group(maintain_averages_op)

    logits_ema = classifier(inp, is_training_pl, getter=get_getter(ema), reuse=True)
    correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
    accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))


    def linear_decay(decay_start, decay_end, epoch):
        return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start), 1)

    # all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print("list des vars")
    # [print(var.name) for var in all_var]
    config = tf.ConfigProto(device_count={'GPU': 0})
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(200)):
            begin = time.time()

            # randomly permuted minibatches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            train_loss = train_acc = test_acc = test_acc_ema = 0

            lr = FLAGS.learning_rate * linear_decay(100, 200, epoch)

            for t in tqdm(range(nr_batches_train)):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate_pl: lr}

                _, ls, acc = sess.run([train_dis_op, loss, accuracy], feed_dict=feed_dict)

                train_loss += ls
                train_acc += acc

            train_loss /= nr_batches_train
            train_acc /= nr_batches_train *100

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}

                acc, acc_ema = sess.run([accuracy, accuracy_ema], feed_dict=feed_dict)
                test_acc += acc
                test_acc_ema += acc_ema

            test_acc /= nr_batches_test
            test_acc_ema /= nr_batches_test


            tqdm.write("Epoch %03d | Time = %03ds | lr = %.3e | loss train = %.4f | train acc = %.2f | test acc = %.2f | test acc_ema = %.2f" %
                       (epoch, time.time() - begin, lr,train_loss, train_acc * 100, test_acc *100, test_acc_ema *100))

            if status_reporter: # report status for ray tune
                status_reporter(timesteps_total=epoch, mean_accuracy=test_acc)


if __name__ == '__main__':
    tf.app.run()


def train_with_dic(config=None, reporter=None):
    global status_reporter
    status_reporter = reporter

    argv = []     # config dic => argv
    for key, value in config.items():
        argv.extend(['--' + key, str(value)])

    tf.app.run(main=main, argv=sys.argv + argv)

    # if config['verbose'] is True:
    #     tf.app.run(main=main, argv=[sys.argv[0]] + argv)
    # else:
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # warning tensorflow consol
    #     with suppress_stdout(): # redirect ouptut to null
    #         tf.app.run(main=main, argv=[sys.argv[0]] + argv)
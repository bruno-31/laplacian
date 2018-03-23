import tensorflow as tf
import numpy as np
from dnn import DNN
from data import cifar10_input
from tqdm import tqdm
import os
import time

from libs.utils import save_images, mkdir
from spectral_gan.net import DCGANGenerator

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_integer('seed', 1546, 'seed[1]')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
flags.DEFINE_float('scale', 1e-5, 'scale perturbation')
flags.DEFINE_float('reg_w', 1e-3, 'weight regularization')


FLAGS = flags.FLAGS


def main(_):
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


    # placeholder model
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate_pl = tf.placeholder(tf.float32, [], name='adam_learning_rate_pl')
    config = FLAGS.__flags
    generator = DCGANGenerator(**config)
    classifier = DNN()

    latent_dim = generator.generate_noise().shape[1]
    z = tf.placeholder(tf.float32, shape=[None, latent_dim])

    x_hat = generator(z, is_training=is_training_pl)
    logits = classifier(inp, is_training=is_training_pl)

    perturb = tf.random_normal([FLAGS.batch_size, latent_dim], mean=0, stddev=0.01)
    z_pert = z + FLAGS.scale * perturb / (tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, latent_dim]))

    x_hat_pert = generator(z_pert, is_training=is_training_pl,reuse=True)
    logits_gen = classifier(x_hat, is_training=is_training_pl,reuse=True)
    logits_gen_perturb = classifier(x_hat_pert, is_training=is_training_pl,reuse=True)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)
    j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))
    loss += FLAGS.reg_w * j_loss

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), lbl)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    g_vars = tf.global_variables(scope='generator')
    dnn_vars = [var for var in tf.trainable_variables() if var not in g_vars]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # control dependencies for batch norm ops
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, var_list=dnn_vars)

    def linear_decay(decay_start, decay_end, epoch):
        return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start), 1)

    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print("list des vars")
    # [print(var.name) for var in all_var]
    # print("name var")
    # [print(var.name for var in g_vars)]

    saver = tf.train.Saver(var_list=g_vars)
    var_init = [var for var in tf.global_variables() if var not in g_vars]
    init_op = tf.variables_initializer(var_list=var_init)
    with tf.Session() as sess:
        sess.run(init_op)
        if tf.train.latest_checkpoint('snapshots') is not None:
            saver.restore(sess, tf.train.latest_checkpoint('snapshots'))
            print("model restored")

        for epoch in tqdm(range(200)):
            begin = time.time()

            # randomly permuted minibatches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            train_loss = train_acc = test_acc = train_j = 0

            lr = FLAGS.learning_rate * linear_decay(100, 200, epoch)

            for t in tqdm(range(nr_batches_train)):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate_pl: lr,
                             z: generator.generate_noise()}

                _, ls, acc, j = sess.run([train_op, loss, accuracy, j_loss], feed_dict=feed_dict)

                train_loss += ls
                train_acc += acc
                train_j += j

            train_loss /= nr_batches_train
            train_acc /= nr_batches_train
            train_j /=nr_batches_train

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False,
                             z: generator.generate_noise()}

                test_acc += sess.run(accuracy, feed_dict=feed_dict)
            test_acc /= nr_batches_test

            tqdm.write("Epoch %03d | Time = %03ds | lr = %.4f | loss train = %.4f | loss j = %.4f | train acc = %.4f | test acc = %.4f" %
                       (epoch, time.time() - begin, lr,train_loss, train_j ,train_acc, test_acc))

if __name__ == '__main__':
    tf.app.run()

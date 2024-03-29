import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data import cifar10_input
# from cifar_gan import discriminator, generator
import sys
from net import DCGANGenerator, SNDCGAN_Discrminator


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 25, "batch size [250]")
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log/cifar', 'log directory')
flags.DEFINE_integer('seed', 10, 'seed numpy')
flags.DEFINE_integer('seed_data', 10, 'seed data')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('ma_decay', 0.9999, 'exp moving average for inference [0.9999]')

flags.DEFINE_float('scale', 1e-5, 'scale perturbation')
flags.DEFINE_float('nabla_w', 1e-3, 'weight regularization')
flags.DEFINE_integer('decay_start', 1200, 'start of learning rate decay')
flags.DEFINE_integer('epoch', 1400, 'labeled data per class')
flags.DEFINE_boolean('nabla', True, 'enable manifold reg')

flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')

flags.DEFINE_integer('freq_print', 100, 'frequency image print tensorboard [10000]')
flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
flags.DEFINE_integer('freq_save', 50, 'frequency saver epoch[50]')

FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def linear_decay(decay_start, decay_end, epoch):
    return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start),1)


def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling

    # load CIFAR-10
    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    # select labeled data
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    config = FLAGS.__flags
    generator = DCGANGenerator(**config)
    discriminator = SNDCGAN_Discrminator(output_dim=10, features=True, **config)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    increase_global_step = global_step.assign(global_step + 1)

    '''construct graph'''
    print('constructing graph')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')

    # scalar pl
    lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    gen_inp = generator(random_z, is_training_pl)
    logits_gen, layer_fake = discriminator(gen_inp, update_collection=None, features=True)
    logits_unl, layer_real = discriminator(unl, update_collection="NO_OPS", features=True)
    logits_lab, _ = discriminator(inp, update_collection="NO_OPS")


    with tf.name_scope('loss_functions'):
        l_unl = tf.reduce_logsumexp(logits_unl, axis=1)
        l_gen = tf.reduce_logsumexp(logits_gen, axis=1)
        # discriminator
        loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab))
        loss_unl = - 0.5 * tf.reduce_mean(l_unl) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))

        # generator
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
        loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab

        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_classifier = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    with tf.name_scope('optimizers'):
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1,
                                           beta2=FLAGS.adam_beta2)

        d_gvs = optimizer.compute_gradients(loss_dis, var_list=d_vars)
        g_gvs = optimizer.compute_gradients(loss_gen, var_list=g_vars)
        d_solver = optimizer.apply_gradients(d_gvs)
        g_solver = optimizer.apply_gradients(g_gvs)

    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
    maintain_averages_op = ema.apply(d_vars)

    with tf.control_dependencies([d_solver]):
        train_dis_op = tf.group(maintain_averages_op)

    logits_ema, _ = discriminator(inp, update_collection="NO_OPS", getter=get_getter(ema))
    correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
    accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))

    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])

        with tf.name_scope('generator'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])

        with tf.name_scope('images'):
            tf.summary.image('gen_images', gen_inp, 10, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
            tf.summary.scalar('learning_rate', lr_pl, ['epoch'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')

    '''//////training //////'''
    print('start training')
    with tf.Session() as sess:
        tf.set_random_seed(rng.randint(2 ** 10))
        sess.run(tf.global_variables_initializer())
        print('\ninitialization done')

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        train_batch = 0

        for epoch in tqdm(range(FLAGS.epoch)):
            begin = time.time()

            train_loss_lab=train_loss_unl=train_loss_gen=train_acc=test_acc=test_acc_ma=train_j_loss = 0
            lr = FLAGS.learning_rate * linear_decay(FLAGS.decay_start,FLAGS.epoch,epoch)

            # construct randomly permuted batches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):  # same size lbl and unlb
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            # training
            for t in tqdm(range(nr_batches_train)):

                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             lr_pl: lr}
                _, acc, lu, lb, sm = sess.run([train_dis_op, accuracy_classifier, loss_lab, loss_unl, sum_op_dis],
                                                  feed_dict=feed_dict)
                train_loss_unl += lu
                train_loss_lab += lb
                train_acc += acc
                if (train_batch % FLAGS.step_print) == 0:
                    writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([g_solver, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True,
                                                                                      lr_pl: lr})
                train_loss_gen += lg
                if (train_batch % FLAGS.step_print) == 0:
                    writer.add_summary(sm, train_batch)

                if (train_batch % FLAGS.freq_print == 0) & (train_batch != 0):
                    ran_from = np.random.randint(0, trainx_unl.shape[0] - FLAGS.batch_size)
                    ran_to = ran_from + FLAGS.batch_size
                    sm = sess.run(sum_op_im,
                                  feed_dict={is_training_pl: True, unl: trainx_unl[ran_from:ran_to]})
                    writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_loss_gen /= nr_batches_train
            train_acc /= nr_batches_train
            train_j_loss /= nr_batches_train

            # Testing moving averaged model and raw model
            if (epoch % FLAGS.freq_test == 0) | (epoch == FLAGS.epoch-1):
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl: testy[ran_from:ran_to],
                                 is_training_pl: False}
                    acc, acc_ema = sess.run([accuracy_classifier, accuracy_ema], feed_dict=feed_dict)
                    test_acc += acc
                    test_acc_ma += acc_ema
                test_acc /= nr_batches_test
                test_acc_ma /= nr_batches_test

                print(
                    "Epoch %d | time = %ds | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                    "| train acc = %.4f| test acc = %.4f | test acc ema = %0.4f"
                    % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc,
                       test_acc, test_acc_ma))

if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf
import numpy as np
from data import cifar10_input
from tqdm import tqdm
import os, time, sys
from spectral_gan.net import DCGANGenerator
from libs.utils import suppress_stdout

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_integer("mc_size", 100, "batch size [128]")
flags.DEFINE_integer("epoch", 200, "batch size [128]")
flags.DEFINE_integer("decay", 100, "batch size [128]")
flags.DEFINE_string('data_dir', '/tmp/data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_string('snapshot', '/home/data/bruno/manifold/snapshots', 'snapshot directory')
flags.DEFINE_float('ma_decay', 0.99, 'exp moving average for inference [0.9999]')
flags.DEFINE_integer('freq_print', 200, 'frequency image print tensorboard [10000]')
flags.DEFINE_integer('freq_test', 5, 'frequency image print tensorboard [10000]')

flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_integer('seed', 1546, 'seed[1]')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
flags.DEFINE_float('scale', 1e-2, 'scale perturbation')
flags.DEFINE_float('reg_w', 1e-4, 'weight regularization')
flags.DEFINE_boolean('verbose', True, 'verbose')

flags.DEFINE_string('grad', 'stochastic', 'choose type of regularisation')
flags.DEFINE_boolean('reg', True, 'enable reg or not')

flags.DEFINE_boolean('tiny_cnn', False, 'verbose')



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
    print("trainx shape:", trainx.shape)


    # placeholder model
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    gan_is_training_pl = tf.placeholder(tf.bool, [], name='gan_is_training_pl')
    learning_rate_pl = tf.placeholder(tf.float16, [], name='adam_learning_rate_pl')

    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    generator = DCGANGenerator(batch_size=FLAGS.mc_size)
    latent_dim = generator.generate_noise().shape[1]
    z = tf.placeholder(tf.float32, shape=[None, latent_dim])

    if not FLAGS.tiny_cnn:
        from dnn import classifier as classifier
    else:
        from dnn import tiny_classifier as classifier

    x_hat = generator(z, is_training=gan_is_training_pl)
    logits = classifier(inp, is_training=is_training_pl)
    logits_gen = classifier(x_hat, is_training=is_training_pl,reuse=True)

    def get_jacobian(y,x):
        with tf.name_scope("jacob"):
            grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)],
                             axis=2)
        return grads

    if FLAGS.grad == 'stochastic':
        print('stochastic reg enabled ...')
        perturb = tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01)
        z_pert = z + FLAGS.scale * perturb / (
                    tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, latent_dim]))
        x_hat_pert = generator(z_pert, is_training=gan_is_training_pl, reuse=True)
        logits_gen_perturb = classifier(x_hat_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))

    elif FLAGS.grad == 'stochastic_v2':
        print('stochastic v2 reg enabled ...')
        perturb = tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01)
        perturb_hat = tf.nn.l2_normalize(perturb,dim=[1])
        x_pert = generator(perturb_hat, is_training=gan_is_training_pl, reuse=True)
        logits_gen_perturb = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))

    elif FLAGS.grad == 'gradient':
        print('stochastic reg enabled ...')
        k=[]
        for j in range(10):
            grad = tf.gradients(logits_gen[:, j], z)
            k.append(grad)
        J = tf.stack(k)
        J = tf.squeeze(J)
        J = tf.transpose(J, perm=[1, 0, 2])  # jacobian
        j_n = tf.reduce_sum(tf.square(J), axis=[1, 2])
        j_loss = tf.reduce_mean(j_n)


    elif FLAGS.grad == 'isotropic_mc':
        print('isotropic mc reg enabled ...')
        epsilon = tf.random_normal([FLAGS.mc_size]+inp.get_shape().as_list()[-3:], mean=0, stddev=0.01) # gaussian noise [mc_size, 32,32,3]
        epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3]) # normalised gaussian noise [mc_size, 32,32,3]
        x_pert = x_hat+FLAGS.scale * epsilon_hat
        logits_gen_pert_iso = classifier(x_pert, is_training=is_training_pl,reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_pert_iso), axis=1))

    elif FLAGS.grad == 'isotropic_inp':
        print('isotropic inp reg enabled ...')
        epsilon = tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
                                   stddev=0.01)  # gaussian noise [mc_size, 32,32,3]
        epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3])  # normalised gaussian noise [mc_size, 32,32,3]
        x_pert = inp + FLAGS.scale * epsilon_hat
        logits_inp_pert_iso = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_inp_pert_iso), axis=1))

    elif FLAGS.grad == 'isotropic_rnd':
        print('isotropic rnd reg enabled ...')
        epsilon = tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
                                   stddev=0.01)  # gaussian noise [mc_size, 32,32,3]
        epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3])  # normalised gaussian noise [mc_size, 32,32,3]
        rnd_img = tf.random_uniform(shape=[FLAGS.mc_size] + inp.get_shape().as_list()[-3:], minval=-1,maxval=1)
        x_pert = rnd_img + FLAGS.scale * epsilon_hat
        logits_inp_pert_iso = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_inp_pert_iso), axis=1))

    ######## loss function #######
    xentropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl)

    if FLAGS.reg:
        print('regularrization enabled')
        loss = xentropy + FLAGS.reg_w * j_loss
    else:
        print('cnn baseline ...')
        loss = xentropy

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), lbl)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # g_vars = tf.global_variables(scope='generator')
    g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
    dnn_vars = [var for var in tf.trainable_variables() if var not in g_vars]

    # [print(var.name) for var in dnn_vars]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # control dependencies for batch norm ops
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss, var_list=dnn_vars)

    dvars = [var for var in tf.trainable_variables() if 'classifier' in var.name]
    # [print(var) for var in dvars]
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
    # print("name var")
    # [print(var.name for var in g_vars)]

    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('xentropy', xentropy, ['dis'])
            tf.summary.scalar('laplacian_loss', j_loss, ['dis'])

        with tf.name_scope('images'):
            tf.summary.image('gen_images', x_hat, 4, ['image'])
            tf.summary.image('gen_pert', x_pert, 4, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
            tf.summary.scalar('learning_rate', learning_rate_pl, ['epoch'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')

    print("batch size monte carlo: ",generator.generate_noise().shape)

    saver = tf.train.Saver(var_list=g_vars)
    var_init = [var for var in tf.global_variables() if var not in g_vars]
    init_op = tf.variables_initializer(var_list=var_init)

    # config = tf.ConfigProto(device_count={'GPU': 0})
    # config.gpu_options.allow_growth = True


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        sess.run(init_op)
        if tf.train.latest_checkpoint(FLAGS.snapshot) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.snapshot))
            print("model restored @ %s" % FLAGS.snapshot)
        train_batch = 0
        for epoch in tqdm(range(FLAGS.epoch),disable=not FLAGS.verbose):
            begin = time.time()

            # randomly permuted minibatches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            train_loss = train_acc = test_acc = train_j = test_acc_ema = 0

            lr = FLAGS.learning_rate * linear_decay(FLAGS.decay, FLAGS.epoch, epoch)

            for t in tqdm(range(nr_batches_train), disable=not FLAGS.verbose):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             gan_is_training_pl: False,
                             learning_rate_pl: lr,
                             z: generator.generate_noise()}

                _, ls, acc, j, sm = sess.run([train_dis_op, loss, accuracy, j_loss,sum_op_dis], feed_dict=feed_dict)

                train_loss += ls
                train_acc += acc
                train_j += j
                writer.add_summary(sm, train_batch)
                train_batch += 1

            train_loss /= nr_batches_train
            train_acc /= nr_batches_train
            train_j /=nr_batches_train

            if (train_batch % FLAGS.freq_print == 0) & (train_batch != 0):
                sm = sess.run(sum_op_im, feed_dict={gan_is_training_pl: False,z: generator.generate_noise(), inp:trainx[:FLAGS.batch_size]})
                writer.add_summary(sm, train_batch)

            if (epoch % FLAGS.freq_test == 0):
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl: testy[ran_from:ran_to],
                                 is_training_pl: False}

                    acc, acc_ema = sess.run([accuracy,accuracy_ema], feed_dict=feed_dict)
                    test_acc += acc
                    test_acc_ema += acc_ema
                test_acc /= nr_batches_test
                test_acc_ema /= nr_batches_test

                sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                        acc_test_pl: test_acc,
                                                        acc_test_pl_ema: test_acc_ema,
                                                        learning_rate_pl:lr})
                writer.add_summary(sum, epoch)

                tqdm.write("Epoch %03d | Time = %03ds | lr = %.3e | loss train = %.4f | train acc = %.2f | test acc = %.2f | test acc_ema = %.2f" %
                           (epoch, time.time() - begin, lr,train_loss, train_acc * 100, test_acc *100, test_acc_ema *100))

                if status_reporter: # report status for ray tune
                    status_reporter(timesteps_total=epoch, mean_accuracy=test_acc_ema)


if __name__ == "__main__":
    tf.app.run()


def train_with_dic(config=None, reporter=None):
    global status_reporter
    status_reporter = reporter
    argv = []     # config dic => argv
    for key, value in config.items():
        argv.extend(['--'+key, str(value)])

    # print("\nParameters:")
    # for attr, value in sorted(config.items()):
    #     print("{}={}".format(attr.lower(), value))
    # print("")

    if config['verbose'] == 'True':
        tf.app.run(main=main, argv=[sys.argv[0]] + argv)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # warning tensorflow consol
        with suppress_stdout(): # redirect ouptut to null
            tf.app.run(main=main, argv=[sys.argv[0]] + argv)



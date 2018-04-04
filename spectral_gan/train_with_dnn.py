import timeit

import numpy as np
import tensorflow as tf

from libs.input_helper import Cifar10
from libs.utils import save_images, mkdir, next_batch
from net import DCGANGenerator, SNDCGAN_Discrminator
import _pickle as pickle
from libs.inception_score.model import get_inception_score
import os
from data import cifar10_input
from lap_dnn.dnn import classifier

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 100000, '')
flags.DEFINE_integer('snapshot_interval', 10000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 10000, 'interval of evaluation')
flags.DEFINE_integer('test_interval', 100, 'interval of evaluation')

flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 1, 'n discrminator train')
flags.DEFINE_string('snapshot', '/tmp/snaphots', 'snapshot directory')
flags.DEFINE_string('data_dir', './tmp/data/cifar-10-python/', 'data directory')
flags.DEFINE_integer('seed', 10, 'seed numpy')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')

flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_float('reg_w', 1e-3, 'weight regularization')

mkdir('tmp')


##############################################
trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
trainx_unl = trainx.copy()
# select labeled data
rng = np.random.RandomState(FLAGS.seed)  # seed labels
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
##############################################
x, y = next_batch(FLAGS.batch_size, trainx, labels=trainy)


INCEPTION_FILENAME = 'inception_score.pkl'
config = FLAGS.__flags
generator = DCGANGenerator(**config)
discriminator = SNDCGAN_Discrminator(**config)
data_set = Cifar10(batch_size=FLAGS.batch_size)

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, update_collection=None)
# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")
# Softplus at the end as in the official code of author at chainer-gan-lib github repository
d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(FLAGS.logdir)

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss, var_list=g_vars)
d_solver = optimizer.apply_gradients(d_gvs)
g_solver = optimizer.apply_gradients(g_gvs)


######################CNN########################
inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input')

logits = classifier(inp, is_training=is_training)
logits_gen = classifier(x_hat, is_training=is_training, reuse=True)

# print(logits_gen)
k = []
for j in range(10):
    grad = tf.gradients(logits_gen[:, j], z)
    # print(grad)
    k.append(grad)
J = tf.stack(k)
J = tf.squeeze(J)
J = tf.transpose(J, perm=[1, 0, 2])  # jacobian
j_n = tf.reduce_sum(tf.square(J), axis=[1, 2])
laplacian = tf.reduce_mean(j_n)

loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=lbl) + FLAGS.reg_w * laplacian

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), lbl)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# dnn_vars = [var for var in tf.global_variables() if 'dnn' in var.name]
dnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

optimizer = tf.train.AdamOptimizer(learning_rate=3e-3)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='classifier')# control dependencies for batch norm ops
# print("update ops")
# [print(op) for op in update_ops]
# print("")
# print("vars dnn")
# [print(var) for var in dnn_vars]
# print("")

with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, var_list=dnn_vars)
#################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))

np.random.seed(1337)
sample_noise = generator.generate_noise() #fixed latent code for visualization
np.random.seed()
iteration = sess.run(global_step)
start = timeit.default_timer()

is_start_iteration = True
inception_scores = []
while iteration < FLAGS.max_iter:
    _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: generator.generate_noise(), is_training: True})
    for _ in range(FLAGS.n_dis):
        _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                             feed_dict={x: next_batch(FLAGS.batch_size,trainx_unl), z: generator.generate_noise(), is_training: True})

        xx,yy = next_batch(FLAGS.batch_size, trainx, labels=trainy)

        feed_dict = {z: generator.generate_noise(), inp: xx, lbl:yy,
                     is_training: True}
        _,acc = sess.run([train_op,accuracy],feed_dict=feed_dict)

    sess.run(increase_global_step)


    if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
        summary_writer.add_summary(summaries, global_step=iteration)
        stop = timeit.default_timer()
        print('Iter {}: d_loss = {:4f}, g_loss = {:4f}, time = {:2f}s'.format(iteration, d_loss_curr, g_loss_curr,
                                                                              stop - start))
        start = stop

    if (iteration + 1) % FLAGS.test_interval == 0 and not is_start_iteration: # compute classifier score
        print("computing score classifier ........")
        test_acc = 0
        for t in range(nr_batches_test):
            ran_from = t * FLAGS.batch_size
            ran_to = (t + 1) * FLAGS.batch_size
            feed_dict = {inp: testx[ran_from:ran_to],
                         lbl: testy[ran_from:ran_to],
                         is_training: False}

            acc = sess.run(accuracy, feed_dict=feed_dict)
            test_acc += acc
        test_acc /= nr_batches_test

        print("test acc = %.2f"%(test_acc))
        print('...............')

    if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration: # saveimages and model
        saver.save(sess, os.path.join(FLAGS.logdir,'model.ckpt'), global_step=iteration)
        sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
        save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))

    if (iteration + 1) % FLAGS.evaluation_interval == 0: # compute inception
        sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
        save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
        # Sample 50000 images for evaluation
        print("Evaluating...")
        num_images_to_eval = 50000
        eval_images = []
        num_batches = num_images_to_eval // FLAGS.batch_size + 1
        print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
        np.random.seed(0)
        for _ in range(num_batches):
            images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
            eval_images.append(images)
        np.random.seed()
        eval_images = np.vstack(eval_images)
        eval_images = eval_images[:num_images_to_eval]
        eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        # Calc Inception score
        eval_images = list(eval_images)
        inception_score_mean, inception_score_std = get_inception_score(eval_images)
        print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))
        inception_scores.append(dict(mean=inception_score_mean, std=inception_score_std))
        with open(INCEPTION_FILENAME, 'wb') as f:
            pickle.dump(inception_scores, f)

    iteration += 1
    is_start_iteration = False

"""
From official improved_gan_training github repository
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import os,sys
import tensorflow as tf
from contextlib import contextmanager


def save_images(X, save_path):
  # [-1, 1] -> [0,255]
  if isinstance(X.flatten()[0], np.floating):
    X = ((X + 1.) * 127.5).astype('uint8')

  n_samples = X.shape[0]
  rows = int(np.sqrt(n_samples))
  while n_samples % rows != 0:
    rows -= 1

  nh, nw = rows, n_samples // rows

  if X.ndim == 2:
    X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

  if X.ndim == 4:
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3))
  elif X.ndim == 3:
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw))

  for n, x in enumerate(X):
    j = n // nw
    i = n % nw
    img[j * h:j * h + h, i * w:i * w + w] = x

  imsave(save_path, img)


def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


import numpy as np

def next_batch(num, data, labels=None):
    '''
    Return a total of `num` random samples and labels.
    '''
    if labels is not None:
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    else:
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        return np.asarray(data_shuffle)

################################################################################

    x_hat = generator(z, is_training=gan_is_training_pl)
    logits = classifier(inp, is_training=is_training_pl)
    logits_gen = classifier(x_hat, is_training=is_training_pl,reuse=True)



def add_regularization(logits, logits_gen, x_hat):
    def get_jacobian(y,x):
        with tf.name_scope("jacob"):
            grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)], axis=2)
        return grads

    if FLAGS.grad == 'stochastic':
        print('stochastic reg enabled ...')
        perturb = tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01)
        z_pert = z + FLAGS.scale * perturb / (
                    tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, latent_dim]))
        x_pert = generator(z_pert, is_training=gan_is_training_pl, reuse=True)
        logits_gen_perturb = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))

        tf.reduce_mean(tf.reduce_sum(tf.square(get_jacobian(logits_gen,z)),axis=[1,2]))


    elif FLAGS.grad == 'stochastic_v2':
        print('stochastic v2 reg enabled ...')
        perturb = tf.nn.l2_normalize(tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01),dim=[1])
        x_pert = generator(z+FLAGS.scale * perturb, is_training=gan_is_training_pl, reuse=True)
        logits_gen_perturb = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))

    elif FLAGS.grad == 'isotropic_mc':
        print('isotropic mc reg enabled ...')
        perturb = tf.nn.l2_normalize(tf.random_normal([FLAGS.mc_size]+inp.get_shape().as_list()[-3:], mean=0, stddev=0.01), dim=[1, 2, 3]) # gaussian noise [mc_size, 32,32,3]
        x_pert = x_hat+FLAGS.scale * perturb
        logits_gen_pert = classifier(x_pert, is_training=is_training_pl,reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_pert), axis=1))

    elif FLAGS.grad == 'isotropic_inp':
        print('isotropic inp reg enabled ...')
        perturb = tf.nn.l2_normalize(tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
                                   stddev=0.01), dim=[1, 2, 3])   # gaussian noise [mc_size, 32,32,3]
        x_pert = inp + FLAGS.scale * perturb
        logits_inp_pert = classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_inp_pert), axis=1))

    elif FLAGS.grad == 'isotropic_rnd':
        print('isotropic rnd reg enabled ...')
        epsilon = tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
                                   stddev=0.01)  # gaussian noise [mc_size, 32,32,3]
        epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3])  # normalised gaussian noise [mc_size, 32,32,3]
        rnd_img = tf.random_uniform(shape=[FLAGS.mc_size] + inp.get_shape().as_list()[-3:], minval=-1,maxval=1)
        x_pert = rnd_img + FLAGS.scale * epsilon_hat
        logits_pert= classifier(x_pert, is_training=is_training_pl, reuse=True)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_pert), axis=1))

    elif FLAGS.grad == 'grad_latent':
        print('grad latent enabled ...')
        grad = get_jacobian(logits_gen, z)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad), axis=[1, 2]))

    elif FLAGS.grad == 'grad_mc':
        print('grad mc enabled ...')
        grad = get_jacobian(logits_gen, x_hat)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad), axis=[1, 2]))

    elif FLAGS.grad == 'grad_inp':
        print('grad inp enabled ...')
        grad = get_jacobian(logits, inp)
        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad), axis=[1, 2]))

    elif FLAGS.grad == 'old_grad':
        print('old grad enabled ...')
        k=[]
        for j in range(10):
            grad = tf.gradients(logits_gen[:, j], z)
            k.append(grad)
        J = tf.stack(k)
        J = tf.squeeze(J)
        J = tf.transpose(J, perm=[1, 0, 2])  # jacobian
        j_n = tf.reduce_sum(tf.square(J), axis=[1, 2])
        j_loss = tf.reduce_mean(j_n)
    return j_loss


#########33
    # def get_jacobian(y,x):
    #     with tf.name_scope("jacob"):
    #         grads = tf.stack([tf.gradients(yi, x)[0] for yi in tf.unstack(y, axis=1)],
    #                          axis=2)
    #     return grads
    #
    #
    # if FLAGS.grad == 'stochastic':
    #     print('stochastic reg enabled ...')
    #     perturb = tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01)
    #     z_pert = z + FLAGS.scale * perturb / (
    #                 tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, latent_dim]))
    #     x_pert = generator(z_pert, is_training=gan_is_training_pl, reuse=True)
    #     logits_gen_perturb = classifier(x_pert, is_training=is_training_pl, reuse=True)
    #     j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))
    #
    # elif FLAGS.grad == 'stochastic_v2':
    #     print('stochastic v2 reg enabled ...')
    #     perturb = tf.random_normal([FLAGS.mc_size, latent_dim], mean=0, stddev=0.01)
    #     perturb_hat = tf.nn.l2_normalize(perturb,dim=[1])
    #     x_pert = generator(perturb_hat, is_training=gan_is_training_pl, reuse=True)
    #     logits_gen_perturb = classifier(x_pert, is_training=is_training_pl, reuse=True)
    #     j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb), axis=1))
    #
    # elif FLAGS.grad == 'gradient':
    #     print('stochastic reg enabled ...')
    #     k=[]
    #     for j in range(10):
    #         grad = tf.gradients(logits_gen[:, j], z)
    #         k.append(grad)
    #     J = tf.stack(k)
    #     J = tf.squeeze(J)
    #     J = tf.transpose(J, perm=[1, 0, 2])  # jacobian
    #     j_n = tf.reduce_sum(tf.square(J), axis=[1, 2])
    #     j_loss = tf.reduce_mean(j_n)
    #
    # elif FLAGS.grad == 'isotropic_mc':
    #     print('isotropic mc reg enabled ...')
    #     epsilon = tf.random_normal([FLAGS.mc_size]+inp.get_shape().as_list()[-3:], mean=0, stddev=0.01) # gaussian noise [mc_size, 32,32,3]
    #     epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3]) # normalised gaussian noise [mc_size, 32,32,3]
    #     x_pert = x_hat+FLAGS.scale * epsilon_hat
    #     logits_gen_pert_iso = classifier(x_pert, is_training=is_training_pl,reuse=True)
    #     j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_pert_iso), axis=1))
    #
    # elif FLAGS.grad == 'isotropic_inp':
    #     print('isotropic inp reg enabled ...')
    #     epsilon = tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
    #                                stddev=0.01)  # gaussian noise [mc_size, 32,32,3]
    #     epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3])  # normalised gaussian noise [mc_size, 32,32,3]
    #     x_pert = inp + FLAGS.scale * epsilon_hat
    #     logits_inp_pert_iso = classifier(x_pert, is_training=is_training_pl, reuse=True)
    #     j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_inp_pert_iso), axis=1))
    #
    # elif FLAGS.grad == 'isotropic_rnd':
    #     print('isotropic rnd reg enabled ...')
    #     epsilon = tf.random_normal([FLAGS.mc_size] + inp.get_shape().as_list()[-3:], mean=0,
    #                                stddev=0.01)  # gaussian noise [mc_size, 32,32,3]
    #     epsilon_hat = tf.nn.l2_normalize(epsilon, dim=[1, 2, 3])  # normalised gaussian noise [mc_size, 32,32,3]
    #     rnd_img = tf.random_uniform(shape=[FLAGS.mc_size] + inp.get_shape().as_list()[-3:], minval=-1,maxval=1)
    #     x_pert = rnd_img + FLAGS.scale * epsilon_hat
    #     logits_inp_pert_iso = classifier(x_pert, is_training=is_training_pl, reuse=True)
    #     j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_inp_pert_iso), axis=1))
    #
    # elif FLAGS.grad == 'test':
    #     grad = get_jacobian(logits_gen, z)
    #     print(grad)
    #     norm = tf.reduce_sum(tf.square(get_jacobian(logits_gen, z)), axis=[1, 2])
    #     print(norm)
    #     mc = tf.reduce_mean(norm)
    #     print(mc)
    #     j_loss = mc
    #     x_pert = x_hat





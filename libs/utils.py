"""
From official improved_gan_training github repository
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import os,sys

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

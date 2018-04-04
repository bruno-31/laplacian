import numpy as np
import tensorflow as tf
from libs.utils import save_images, mkdir
from spectral_gan.net import DCGANGenerator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')

mkdir('tmp')
VIZ_VAR = False

config = FLAGS.__flags
generator = DCGANGenerator(**config)

is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

g_vars = tf.global_variables(scope='generator')
saver = tf.train.Saver(var_list=g_vars)
var_init = [var for var in tf.global_variables() if var not in g_vars]
init_op = tf.variables_initializer(var_list=var_init)

if VIZ_VAR:
    [print(var.name) for var in tf.global_variables()]
    print('')
    [print(var.name) for var in var_init]
    print('')
    [print(var.name) for var in g_vars]

# all_var = tf.global_variables()
# print("list des vars")
# [print(var.name) for var in all_var]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(init_op)

if tf.train.latest_checkpoint('snapshots') is not None:
    saver.restore(sess, tf.train.latest_checkpoint('snapshots'))
    print('model restored')


np.random.seed(1337)
print('writing images ..')
for iteration in range(10):
    print('%d/10'%(iteration+1))
    sample_images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
    save_images(sample_images, 'tmp/p{:06d}.png'.format(iteration))



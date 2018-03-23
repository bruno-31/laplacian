from libs.ops import *


class DNN(object):

    def __init__(self, batch_size=64, hidden_activation=tf.nn.relu, output_dim=10, scope='dnn', **kwargs):
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.scope = scope

    def __call__(self, x, is_training,reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            if scope_has_variables(scope):
                scope.reuse_variables()
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_0'), is_training=is_training,
                           name='bn0_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_1'), is_training=is_training,
                           name='bn0_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_2', strides=[2, 2]),
                           is_training=is_training, name='bn0_2'))

            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_0'), is_training=is_training,
                           name='bn1_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_1'), is_training=is_training,
                           name='bn1_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_2', strides=[2, 2]),
                           is_training=is_training, name='bn1_2'))

            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c2_0'), is_training=is_training,
                           name='bn2_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 1, padding='VALID', name='c2_1'), is_training=is_training,
                           name='bn2_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 192, 1, padding='VALID', name='c2_2'), is_training=is_training,
                           name='bn2_2'))
            x = tf.squeeze(tf.layers.average_pooling2d(x, x.get_shape().as_list()[1], strides=1))
            x = tf.layers.dense(x, self.output_dim)
            return x

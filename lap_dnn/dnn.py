from libs.ops import *


class DNN(object):

    def __init__(self, batch_size=64, hidden_activation=tf.nn.relu, output_dim=10, scope='dnn', **kwargs):
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.scope = scope

    # def __call__(self, x, is_training,reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
    #     with tf.variable_scope(self.scope) as scope:
    #         if scope_has_variables(scope):
    #             scope.reuse_variables()
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_0'), is_training=is_training,
    #                        name='bn0_0'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_1'), is_training=is_training,
    #                        name='bn0_1'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 96, 3, padding='SAME', name='c0_2', strides=[2, 2]),
    #                        is_training=is_training, name='bn0_2'))
    #
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_0'), is_training=is_training,
    #                        name='bn1_0'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_1'), is_training=is_training,
    #                        name='bn1_1'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c1_2', strides=[2, 2]),
    #                        is_training=is_training, name='bn1_2'))
    #
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 3, padding='SAME', name='c2_0'), is_training=is_training,
    #                        name='bn2_0'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 1, padding='VALID', name='c2_1'), is_training=is_training,
    #                        name='bn2_1'))
    #         x = self.hidden_activation(
    #             batch_norm(tf.layers.conv2d(x, 192, 1, padding='VALID', name='c2_2'), is_training=is_training,
    #                        name='bn2_2'))
    #         x = tf.squeeze(tf.layers.average_pooling2d(x, x.get_shape().as_list()[1], strides=1))
    #         x = tf.layers.dense(x, self.output_dim)
    #         return x

    def __call__(self, x, is_training,getter=None, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
        with tf.variable_scope(self.scope, custom_getter=getter) as scope:
            if scope_has_variables(scope):
                scope.reuse_variables()
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 64, 3, padding='SAME', name='c0_0'), is_training=is_training,
                           name='bn0_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 64, 3, padding='SAME', name='c0_1'), is_training=is_training,
                           name='bn0_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 64, 3, padding='SAME', name='c0_2', strides=[2, 2]),
                           is_training=is_training, name='bn0_2'))

            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 3, padding='SAME', name='c1_0'), is_training=is_training,
                           name='bn1_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 3, padding='SAME', name='c1_1'), is_training=is_training,
                           name='bn1_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 3, padding='SAME', name='c1_2', strides=[2, 2]),
                           is_training=is_training, name='bn1_2'))

            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 3, padding='SAME', name='c2_0'), is_training=is_training,
                           name='bn2_0'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 1, padding='VALID', name='c2_1'), is_training=is_training,
                           name='bn2_1'))
            x = self.hidden_activation(
                batch_norm(tf.layers.conv2d(x, 128, 1, padding='VALID', name='c2_2'), is_training=is_training,
                           name='bn2_2'))
            x = tf.squeeze(tf.layers.average_pooling2d(x, x.get_shape().as_list()[1], strides=1))
            x = tf.layers.dense(x, self.output_dim)
            return x


def activation(x):
    return leakyReLu(x,alpha=0.1)

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def classifier(inp, is_training, reuse=False, getter=None):
    with tf.variable_scope('classifier', reuse=reuse, custom_getter=getter):
        x = tf.reshape(inp, [-1, 32, 32, 3])

        x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME',strides=2), training=is_training))

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 3, padding='SAME',strides=2), training=is_training))

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 1, padding='VALID'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 128, 1, padding='VALID'), training=is_training))
        # print(x)
        x = tf.squeeze(tf.layers.average_pooling2d(x, x.get_shape().as_list()[1], strides=1))
        # print(x)

        x = tf.layers.dense(x,10)
        return x

def tiny_classifier(inp, is_training, reuse=False, getter=None):
    with tf.variable_scope('classifier', reuse=reuse, custom_getter=getter):
        x = tf.reshape(inp, [-1, 32, 32, 3])

        x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 32, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 32, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 32, 3, padding='SAME',strides=2), training=is_training))

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME',strides=2), training=is_training))

        x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')

        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 3, padding='SAME'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 1, padding='VALID'), training=is_training))
        x = activation(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 1, padding='VALID'), training=is_training))
        # print(x)
        x = tf.squeeze(tf.layers.average_pooling2d(x, x.get_shape().as_list()[1], strides=1))
        # print(x)

        x = tf.layers.dense(x,10)
        return x

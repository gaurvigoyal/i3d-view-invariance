from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from i3d_end import Unit3D

def exp6(inception3d, net, end_points, inputs, is_training, dropout_keep_prob):
  end_point = 'Logits'
  with tf.variable_scope((end_point)):
    net = tf.layers.flatten(net, name = 'Flatten')
    net = tf.nn.dropout(net, dropout_keep_prob)
    net = tf.layers.dense(net,inception3d._num_classes,
      activation = None,
      # activation = tf.nn.relu,
      use_bias = True,
      trainable = is_training,
      name = 'FullyConnected2')
    bn = snt.BatchNorm()
    net = bn(net, is_training=is_training, test_local_stats=False)

  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points

def exp4(inception3d, net, end_points, inputs, is_training, dropout_keep_prob):
  end_point = 'Mixed_5c'
  with tf.variable_scope(end_point):
    with tf.variable_scope('Branch_0'):
      branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=is_training)
    with tf.variable_scope('Branch_1'):
      branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=False)
      branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_1,
                                              is_training=is_training)
    with tf.variable_scope('Branch_2'):
      branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=False)
      branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_2,
                                              is_training=is_training)
    with tf.variable_scope('Branch_3'):
      branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                  strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                  name='MaxPool3d_0a_3x3')
      branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                        name='Conv3d_0b_1x1')(branch_3,
                                              is_training=is_training)
    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points

  end_point = 'Logits'
  with tf.variable_scope(end_point):
    net = tf.nn.avg_pool3d(
      net, 
      ksize=[1, 2, 7, 7, 1],
      strides=[1, 1, 1, 1, 1], 
      padding=snt.VALID)
    net = tf.nn.dropout(net, dropout_keep_prob)
    net = Unit3D(
      output_channels=inception3d._num_classes,
      kernel_shape=[1, 1, 1],
      activation_fn=None,
      use_batch_norm=False,
      use_bias=True,
      name='Conv3d_0c_1x1')(net, is_training=is_training)
    net = tf.squeeze(net, [2, 3], name='SpatialSqueeze')
    net = tf.reduce_mean(net, axis=1)
  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points

def exp5(inception3d, net, end_points, inputs, is_training, dropout_keep_prob):

  end_point = 'Mixed_5c'
  with tf.variable_scope(end_point):
    with tf.variable_scope('Branch_0'):
      branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=is_training)
    with tf.variable_scope('Branch_1'):
      branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=False)
      branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_1,
                                              is_training=is_training)
    with tf.variable_scope('Branch_2'):
      branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=False)
      branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_2,
                                              is_training=is_training)
    with tf.variable_scope('Branch_3'):
      branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                  strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                  name='MaxPool3d_0a_3x3')
      branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                        name='Conv3d_0b_1x1')(branch_3,
                                              is_training=is_training)
    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points

  end_point = 'Logits'
  with tf.variable_scope(end_point):
    # net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
    #                          strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    net = tf.layers.flatten(net, name = 'Flatten')
    net = tf.nn.dropout(net, dropout_keep_prob)
    net = tf.layers.dense(net,inception3d._num_classes,
        activation = None,
        # activation = tf.nn.relu,
        use_bias = True,
        trainable = is_training,
        name = 'FullyConnected')
    bn = snt.BatchNorm()
    net = bn(net, is_training=is_training, test_local_stats=False)                
  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points

def exp7(inception3d, net, end_points, inputs, is_training, dropout_keep_prob):
  end_point = 'Logits'
  with tf.variable_scope(end_point):
    net = tf.layers.flatten(net, name = 'Flatten')
    net = tf.layers.dense(net, 4096,
      activation = tf.nn.relu,
      use_bias = True,
      trainable = is_training,
      name = 'FullyConnected1')
    bn = snt.BatchNorm()
    net = bn(net, is_training=is_training, test_local_stats=False)

    net = tf.nn.dropout(net, dropout_keep_prob)
    net = tf.layers.dense(net,inception3d._num_classes,
      activation = None,
      use_bias = True,
      trainable = is_training,
      name = 'FullyConnected2')
    bn = snt.BatchNorm()
    net = bn(net, is_training=is_training, test_local_stats=False)
  end_points[end_point] = net
  if inception3d._final_endpoint == end_point: return net, end_points


def exp8(inception3d, net, end_points, inputs, is_training, dropout_keep_prob):

  end_point = 'Mixed_5c'
  with tf.variable_scope(end_point):
    with tf.variable_scope('Branch_0'):
      branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=is_training)
    with tf.variable_scope('Branch_1'):
      branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=is_training)
      branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_1,
                                              is_training=is_training)
    with tf.variable_scope('Branch_2'):
      branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(net, is_training=is_training)
      branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_2,
                                              is_training=is_training)
    with tf.variable_scope('Branch_3'):
      branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                  strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                  name='MaxPool3d_0a_3x3')
      branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                        name='Conv3d_0b_1x1')(branch_3,
                                              is_training=is_training)
    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
  end_points[end_point] = net
  if self._final_endpoint == end_point: return net, end_points

  end_point = 'Logits'
  with tf.variable_scope(end_point):
    net = tf.layers.flatten(net, name = 'Flatten')
    net = tf.layers.dense(net, 4096,
      activation = tf.nn.relu,
      use_bias = True,
      trainable = is_training,
      name = 'FullyConnected1')
    bn = snt.BatchNorm()
    net = bn(net, is_training=is_training, test_local_stats=False)
    # net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                           # strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    net = tf.nn.dropout(net, dropout_keep_prob)
    net = tf.layers.dense(net,self._num_classes,
      activation = None,
      use_bias = True,
      trainable = is_training,
      name = 'FullyConnected2')
  if inception3d._final_endpoint == end_point: return net, end_points

# This script is to be used once for a specific model. It takes the I3D model 
# and it's checkpoints, modifies the model, and saves the correct format of a 
# checkpoint, for your custom model.

# Author: Gaurvi Goyal
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import random
import os,sys
import time
import utils

import i3d

batch_size = 15

_IMAGE_SIZE = 224

_CHECKPOINT_PATHS = {
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset', 'None', 'ixmas or nucla')
tf.flags.DEFINE_string('frames', '60', 'number of frames')
tf.flags.DEFINE_string('eval_type', 'flow', 'rgb or flow')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  _VIDEO_FRAMES = int(FLAGS.frames)
  eval_type = FLAGS.eval_type
  dataset = str(FLAGS.dataset)
  complete_variable_map = {}

  try:
    dataset_fullname = {'ixmas': 'ixmas','nucla':'multiview_action_videos'}[dataset]
  except KeyError:
    print('Please provide correct dataset')
    sys.exit(1)

  _NUM_CLASSES = int({'ixmas': '11', 'nucla':'10'}[dataset])

  if eval_type == 'rgb':
    # RGB input has 3 channels.
    model_input = tf.placeholder(tf.float32,
        shape=(batch_size, _VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    y = tf.placeholder(tf.int32, batch_size)
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=False, final_endpoint='Mixed_5b')
      rgb_net, _ = rgb_model( model_input, is_training=False, dropout_keep_prob=1.0)
      
      # Adding a replacement logits layer
      end_point = 'Logits'
      with tf.variable_scope('inception_i3d/'+end_point):
        # rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                               # strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        # if TRAINING:
        #     flow_net = tf.nn.dropout(flow_net, 0.7)
        rgb_net = tf.layers.flatten(rgb_net, name = 'Flatten')
        rgb_net = tf.layers.dense(rgb_net, 4096,
          activation = tf.nn.relu,
          use_bias = True,
          trainable = True,
          name = 'FullyConnected1')
        rgb_logits = tf.layers.dense(rgb_net, _NUM_CLASSES,
          activation = tf.nn.relu,
          use_bias = True,
          trainable = True,
          name = 'FullyConnected2')
     # with tf.variable_scope(end_point):
      #   rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
      #                          strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      #   if TRAINING:
      #       rgb_net = tf.nn.dropout(rgb_net, 0.7)
      #   logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
      #                   kernel_shape=[1, 1, 1],
      #                   activation_fn=None,
      #                   use_batch_norm=False,
      #                   use_bias=True,
      #                   name='Conv3d_0c_1x1')(rgb_net, is_training=True)

      #   logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
      #   rgb_logits = tf.reduce_mean(logits, axis=1)
        
    rgb_variable_map = {}
    rgb_variable_map_to_initialize = []
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        complete_variable_map[variable.name.replace(':0', '')] = variable
        if variable.name.split('/')[2] == 'Logits': 
          rgb_variable_map_to_initialize.append(variable)
          print('===variable_new:', variable)
          continue
        rgb_variable_map[variable.name.replace(':0', '')] = variable
        print('===variable:', variable)
    
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type == 'flow':
    # Flow input has only 2 channels.
    model_input = tf.placeholder(tf.float32,
        shape=(batch_size, _VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    y = tf.placeholder(tf.int32, batch_size)
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5b')
      flow_net, _ = flow_model(model_input, is_training=False, dropout_keep_prob=1.0)

      # Adding a replacement logits layer
      end_point = 'Logits'
      with tf.variable_scope('inception_i3d/'+ end_point):
        flow_net = tf.layers.flatten(flow_net, name = 'Flatten')
        flow_net = tf.layers.dense(flow_net, 4096,
          activation = tf.nn.relu,
          use_bias = True,
          trainable = True,
          name = 'FullyConnected1')
        # flow_net = tf.nn.avg_pool3d(flow_net, ksize=[1, 2, 7, 7, 1],
                               # strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        # if TRAINING:
        #     flow_net = tf.nn.dropout(flow_net, 0.7)
        flow_logits = tf.layers.dense(flow_net, _NUM_CLASSES,
          activation = tf.nn.relu,
          use_bias = True,
          trainable = True,
          name = 'FullyConnected2')
        # logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
        #                 kernel_shape=[1, 1, 1],
        #                 activation_fn=None,
        #                 use_batch_norm=False,
        #                 use_bias=True,
        #                 name='Conv3d_0c_1x1')(flow_net, is_training=True)

        # logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        # flow_logits = tf.reduce_mean(logits, axis=1)
    flow_variable_map = {}
    flow_variable_map_to_initialize = []
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        complete_variable_map[variable.name.replace(':0', '')] = variable
        if variable.name.split('/')[2] == 'Logits': 
          flow_variable_map_to_initialize.append(variable)
          print('===variable_new:', variable)
          continue
        flow_variable_map[variable.name.replace(':0', '')] = variable
        print('===variable:', variable)

    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  
  model_saver = tf.train.Saver(var_list=complete_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    if eval_type == 'rgb':
      rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      for variable in rgb_variable_map_to_initialize:
        sess.run(variable.initializer)
      # sess.run(rgb_variable_map_to_initialize[1].initializer)
      # print(type(rgb_variable_map_to_initialize[0]))
      # print(tf.is_variable_initialized(rgb_variable_map_to_initialize[0]))
    if eval_type =='flow':
      flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      for variable in flow_variable_map_to_initialize:
        sess.run(variable.initializer)
      # sess.run(flow_variable_map_to_initialize[1].initializer)

    utils.ensure_dir('../results/checkpoints/{}7/model_firstckpt_{}'.format(dataset,eval_type))
    save_path = model_saver.save(sess,'../results/checkpoints/{}7/model_firstckpt_{}/model.ckpt'.format(dataset,eval_type))

if __name__ == '__main__':
  tf.app.run(main)
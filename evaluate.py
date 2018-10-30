

"""Loads a file of video names and classifies using a trained model checkpoint."""

import numpy as np
import sonnet as snt
import tensorflow as tf
import random
import os,sys
import time

import i3d_end
import classifiers
from utils import next_feature_batch,get_dataset_info,ensure_dir, load_a_feature
import xlwt

_WRITE = 1
TRAINING = True

FLAGS = tf.flags.FLAGS
initial_step=1
# action_labels = {'a01':0,'a02':1,'a03':2,'a04':3,
#   'a05':4,'a06':5,'a08':6,'a09':7,'a11':8,'a12':9}

tf.flags.DEFINE_string('eval_type', 'flow', 'rgb, flow, or joint')
# tf.flags.DEFINE_string('learning_rate', '0.01', 'Learning Rate')
# tf.flags.DEFINE_string('initial_epoch', '1', 'Initial epoch number')
tf.flags.DEFINE_string('epoch', '0', 'Epoch for which evaluation to be done')
tf.flags.DEFINE_string('graph', '00', 'Graph number')
tf.flags.DEFINE_string('dataset', None, 'nucla or ixmas')
# tf.flags.DEFINE_string('batch_size', '10', 'Batch size')
tf.flags.DEFINE_string('split', 'cam0', 'Validation split')
tf.flags.DEFINE_string('classifier', None, 'classifier to use')
tf.flags.DEFINE_string('feat_frame','8', 'Number of feature frames to use')


# _LABEL_MAP_PATH = 'data/label_map.txt'

def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  # learning_rate = float(FLAGS.learning_rate)
  # initial_epoch = int(FLAGS.initial_epoch)
  epoch = int(FLAGS.epoch)
  # batch_size = int(FLAGS.batch_size)
  _graph_name = FLAGS.graph
  split = FLAGS.split
  classifier = str(FLAGS.classifier) if FLAGS.classifier else None
  frame_dim  = int(FLAGS.feat_frame)

  if classifier is None:
    print('You must specify a \'classifier\'.')
    sys.exit(1)
  try:
    classifier_fn = getattr(classifiers, 'exp{}'.format(classifier[-1:]));
  except AttributeError:
    print('Classifier \'{}\' is not available'.format(classifier))
    sys.exit(1)
  
  try:
    dataset = {'ixmas': 'ixmas','nucla':'multiview_action_videos'}[classifier[:-1]]
  except KeyError:
    print('Dataset could not be understood')
    sys.exit(1)
  if epoch == 0:
  	print("Epoch required.")


  _CHECKPOINT_PATHS = {
    'rgb': '../results/checkpoints/{}/{}_{}_{}/model.ckpt'.format(classifier,eval_type,split,_graph_name),
    'flow': '../results/checkpoints/{}/{}_{}_{}/Epoch_{}.ckpt'.format(classifier,eval_type,split,_graph_name,epoch)}
  # dataset_root = '../dataset/{}/mixed_5b_features/'.format(dataset)
  _NUM_CLASSES = get_dataset_info(dataset)['num_classes']
  action_labels = get_dataset_info(dataset)['action_labels']
  # classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  ensure_dir('../results/evals/{}/'.format(classifier))
  # result_file = '../results/out/{}/{}_{}.txt'.format(classifier,split,epoch)
  result_spreadsheet = '../results/evals/{}/{}_{}_{}.xlwt'.format(classifier,split,_graph_name,epoch)
  
  ####################################
  ########### Model Setup ############
  ####################################
  complete_variable_map = {}
  
  if eval_type in ['rgb', 'joint']:
    
    model_input = tf.placeholder(tf.float32, 
                  shape=(1, frame_dim, 7, 7, 832))
    y = tf.placeholder(tf.int32, 1)
    with tf.variable_scope('RGB'):
      rgb_model = i3d_end.InceptionI3d(
        classifier_fn,
        _NUM_CLASSES, 
        spatial_squeeze=True, 
        final_endpoint='Logits')
      rgb_net, _ = rgb_model(
        model_input, 
        is_training=False, 
        dropout_keep_prob=0.5)
    model_logits = rgb_net

    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        complete_variable_map[variable.name.replace(':0', '')] = variable
        rgb_variable_map[variable.name.replace(':0', '')] = variable
        print('===variable:', variable)
    
    model_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    model_input = tf.placeholder(tf.float32, 
                  shape=(1, frame_dim, 7, 7, 832))
    # y = tf.placeholder(tf.int32, 1)
    with tf.variable_scope('Flow'):
      flow_model = i3d_end.InceptionI3d(
        classifier_fn,
        _NUM_CLASSES, 
        spatial_squeeze=True, 
        final_endpoint='Logits')
      flow_net, _ = flow_model(
        model_input, 
        is_training=False, 
        dropout_keep_prob=0.5)
    model_logits = flow_net

    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        complete_variable_map[variable.name.replace(':0', '')] = variable
        flow_variable_map[variable.name.replace(':0', '')] = variable
        print('===variable:', variable)
    
    model_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  model_predictions = tf.nn.softmax(model_logits)
  
  dataset_split = '../dataset/{}/splits/flow_{}_val_split.txt'.format(dataset,split)
  with open(dataset_split) as f:
    content = f.readlines()
  files = [file.rstrip('\n') for file in content]

    # # sheet.write(r, c, <text>)
  with tf.Session() as sess:
    model_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
    book = xlwt.Workbook()
    sheet = book.add_sheet('Sheet 1')
    count = 0
    acc = np.zeros(len(files))
    sheet.write(count, 0, 'File')
    sheet.write(count, 1, 'Original Label')
    sheet.write(count, 2, 'Predicted label')
    for file in files:
      tf.logging.info(file)
      feed_dict = {}
      feature,label = load_a_feature(file,dataset,frame_dim = 0)
      tf.logging.info('Data loaded, shape=%s', str(feature.shape))
      frames = feature.shape[1]
      pred = np.float32(np.zeros((_NUM_CLASSES,2)))
      for i in range(frames-7):
        out_predictions,out_logits = sess.run([model_predictions,model_logits],feed_dict={model_input:feature[:,i:i+8,:,:,:]})
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]
        pred[sorted_indices[0],0] = np.add(pred[sorted_indices[0],0],1)
        pred[sorted_indices[0],1] = np.add(pred[sorted_indices[0],1], out_predictions[sorted_indices[0]])

      # print(np.argmax(pred,axis=0)[1])
      action_predicted = (list(action_labels.keys())[int(np.argmax(pred,axis=0)[1])])
      # print('File: {} \nAction Predicted: {}\n'.format(label,action_predicted))
      # print(int(label == action_predicted))
      acc[count] = int(label == action_predicted)
      count +=1
      sheet.write(count,0,file)
      sheet.write(count, 1, label)
      sheet.write(count, 2, action_predicted)
    accuracy = np.mean(np.float32(acc))
    print(accuracy)
    book.save(result_spreadsheet)

if __name__ == '__main__':
  tf.app.run(main)

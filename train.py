from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import random
import os,sys
import time

import i3d_end
import classifiers
from utils import next_feature_batch,get_dataset_info,ensure_dir


_WRITE = 1
TRAINING = True

FLAGS = tf.flags.FLAGS
initial_step=1
# action_labels = {'a01':0,'a02':1,'a03':2,'a04':3,
#   'a05':4,'a06':5,'a08':6,'a09':7,'a11':8,'a12':9}

tf.flags.DEFINE_string('eval_type', 'flow', 'rgb, flow, or joint')
tf.flags.DEFINE_string('learning_rate', '0.01', 'Learning Rate')
tf.flags.DEFINE_string('initial_epoch', '1', 'Initial epoch number')
tf.flags.DEFINE_string('epoch', '20', 'Total epochs')
tf.flags.DEFINE_string('graph', '00', 'Graph number')
# tf.flags.DEFINE_string('dataset', None, 'nucla or ixmas')
tf.flags.DEFINE_string('batch_size', '10', 'Batch size')
tf.flags.DEFINE_string('split', 'cam0', 'Validation split')
tf.flags.DEFINE_string('classifier', None, 'classifier to use')
tf.flags.DEFINE_string('feat_frame','8', 'Number of feature frames to use')

def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  learning_rate = float(FLAGS.learning_rate)
  initial_epoch = int(FLAGS.initial_epoch)
  epoch = int(FLAGS.epoch)
  batch_size = int(FLAGS.batch_size)
  _graph_num = FLAGS.graph
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


  _CHECKPOINT_PATHS = {
    'rgb': '../results/checkpoints/{}/model_firstckpt_rgb/model.ckpt'.format(classifier),
    'flow': '../results/checkpoints/{}/model_firstckpt_flow/model.ckpt'.format(classifier)}
  dataset_root = '../dataset/{}/mixed_5b_features/'.format(dataset)
  _NUM_CLASSES = get_dataset_info(dataset)['num_classes']
  
  if initial_epoch > 2:
    _CHECKPOINT_PATHS['flow'] = ('../results/checkpoints/{}/flow_{}_{}/Epoch_{}.ckpt'.format(classifier,split,_graph_num,initial_epoch-1)) 
  
  ensure_dir('../results/out/{}/'.format(classifier))
  result_file = '../results/out/{}/result.txt'.format(classifier)
  print('Graph: {}'.format(_graph_num))    
  print('Learning rate: {}'.format(learning_rate))
  print('Starting epoch: {}'.format(initial_epoch))
  print('Batch size: {}'.format(batch_size))
  print('Validation Split: {}\n'.format(split))
  with open(result_file, 'a') as f:
    f.write('Graph: {}\n'.format(_graph_num))    
    f.write('Learning rate: {}\n'.format(learning_rate))
    f.write('Starting epoch: {}\n'.format(initial_epoch))
    f.write('Batch size: {}\n'.format(batch_size))
    f.write('Validation Split: {}\n'.format(split))
  complete_variable_map = {}

  ####################################
  ########### Model Setup ############
  ####################################

  if eval_type in ['rgb', 'joint']:
    
    model_input = tf.placeholder(tf.float32, 
                  shape=(batch_size, frame_dim, 7, 7, 832))
    y = tf.placeholder(tf.int32, batch_size)
    with tf.variable_scope('RGB'):
      rgb_model = i3d_end.InceptionI3d(
        classifier_fn,
        _NUM_CLASSES, 
        spatial_squeeze=True, 
        final_endpoint='Logits')
      rgb_net, _ = rgb_model(
        model_input, 
        is_training=True, 
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
                  shape=(batch_size, frame_dim, 7, 7, 832))
    y = tf.placeholder(tf.int32, batch_size)
    with tf.variable_scope('Flow'):
      flow_model = i3d_end.InceptionI3d(
        classifier_fn,
        _NUM_CLASSES, 
        spatial_squeeze=True, 
        final_endpoint='Logits')
      flow_net, _ = flow_model(
        model_input, 
        is_training=True, 
        dropout_keep_prob=0.8)
    model_logits = flow_net

    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        complete_variable_map[variable.name.replace(':0', '')] = variable
        flow_variable_map[variable.name.replace(':0', '')] = variable
        print('===variable:', variable)
    
    model_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
  
  ####################################
  ###### Model Evaluation ############
  ####################################
 
 
  model_predictions = tf.nn.softmax(model_logits)
  print( '===model_predictions.shape:', model_predictions.shape)
  loss = tf.losses.sparse_softmax_cross_entropy(logits=model_logits, labels=y)
  loss = tf.reduce_mean(loss)
  loss_ret = tf.cast(loss,tf.float32)
  print('===loss.shape:',     loss.shape)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
  top1_result = tf.nn.in_top_k(model_predictions,y,1)
  top2_result = tf.nn.in_top_k(model_predictions,y,2)
  top1_accuracy = tf.reduce_mean(tf.cast(top1_result,tf.float32))
  top2_accuracy = tf.reduce_mean(tf.cast(top2_result,tf.float32))

  if _WRITE == 1:
    tf.summary.scalar("loss", loss)
    tf.summary.scalar('top1_accuracy',top1_accuracy)
    tf.summary.scalar('top2_accuracy',top2_accuracy)
  ensure_dir('../results/graphs/{}/'.format(classifier))
  _graph_filename_train_total = ('../results/graphs/{}/{}_train_{}/'.format(classifier,split,_graph_num))
  _graph_filename_val_total = ('../results/graphs/{}/{}_val_{}/'.format(classifier,split,_graph_num))

  dataset_split = {'train':'../dataset/{}/splits/flow_{}_train_split.txt'.format(dataset,split), 
    'val': '../dataset/{}/splits/flow_{}_val_split.txt'.format(dataset,split)}

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  ############################
  ######### Session ##########
  ############################

  with tf.Session(config=config) as sess:

    if _WRITE == 1:
      writer_train_total = tf.summary.FileWriter(_graph_filename_train_total,sess.graph)
      writer_train_total.add_graph(tf.get_default_graph())

      writer_val_total = tf.summary.FileWriter(_graph_filename_val_total,sess.graph)
      writer_val_total.add_graph(tf.get_default_graph())
    
    merge_summaries = tf.summary.merge_all()
    # try:
    #   model_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
    # except Exception:
    #   print('Checkpoint File not found. Skipping this test.')
    #   sys.exit(1)

    model_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      
    ###################################
    ############ Training #############
    ###################################
    
    beginning = True
    train_loss = []
    step = initial_step
    current_epoch =initial_epoch
    file_num = 0
    start = time.time()
    initial_start = start
    # steps_in_epoch = 1
    best_accuracies = {'train': 0, 'train_epoch': 0, 'val': 0, 'val_epoch': 0}

    while current_epoch<= epoch: # step < training_iter:
      start = time.time()
      batch_xs, batch_ys, file_num, epoch_completed = next_feature_batch(batch_size, file_num, dataset_split['train'],dataset=dataset, frame_dim = frame_dim)
      # print('Size of the input file is:{}'.format(batch_xs.shape))
      if epoch_completed == False and beginning == False:
        if step%10 == 0:
          print('Batch loaded for step {}.'.format(step))
        # print('Running training...')
        pred,_ = sess.run([model_predictions,optimizer],feed_dict={model_input:batch_xs, y: batch_ys})
        step +=1
        # steps_in_epoch +=1
      else:
        if beginning == False:
          print('EPOCH {} COMPLETED.'.format(current_epoch))
          ensure_dir('../results/checkpoints/{}/{}_{}_{}/'.format(classifier,eval_type,split,_graph_num))
          save_path = model_saver.save(sess,'../results/checkpoints/{}/{}_{}_{}/Epoch_{}.ckpt'.format(classifier,eval_type,split,_graph_num,current_epoch))
        # save_path = model_saver.save(sess,'checkpoints/'+eval_type+'_'+split+'_'+str(_graph_num)+'/Epoch_'+str(current_epoch)+'.ckpt')
          # print('Total time taken from the beginning of the program: {} seconds'.format(time.time()-initial_start))
          current_epoch +=1
        else:
          beginning = False
        if current_epoch == initial_epoch+1:
          with open(result_file, 'a') as f:
            f.write('First Epoch completed in {} seconds.\n'.format(time.time()-initial_start))
        # steps_in_epoch = 1

    ############################################
	  ############# Validation ###################
	  ############################################
      
        file_num=0
        tester=1
        file_num_val =0
        top1_result_vector_train = []
        top2_result_vector_train = []
        top1_accuracy_total = 0
        top2_accuracy_total = 0
        loss_all = []
        start = time.time()
        print('Checking the training set..')

      ########## Testing the training set ########
        while tester == 1:
          batch_xs, batch_ys, file_num_val, val_completed = next_feature_batch(batch_size, file_num_val, dataset_split['train'],dataset=dataset, frame_dim = frame_dim)
          if val_completed ==1:
            tester=0
            file_num_val =0
            break
          [top1_result,top2_result,loss] = sess.run([top1_accuracy,top2_accuracy,loss_ret],feed_dict={model_input:batch_xs, y: batch_ys})
          if top1_result_vector_train == []:
            top1_result_vector_train = top1_result
            top2_result_vector_train = top2_result
            loss_all = loss 
          else:
            top1_result_vector_train=np.hstack((top1_result_vector_train,top1_result))
            top2_result_vector_train=np.hstack((top2_result_vector_train,top2_result))
            loss_all = np.append(loss_all,loss)
        top1_accuracy_total = np.mean(top1_result_vector_train)
        top2_accuracy_total = np.mean(top2_result_vector_train)
        loss_total = np.mean(loss_all)
        # print("Training set Check: The total accuracy for top 1 is {}, for top 2 is {} and for top 3 is {}".format(top1_accuracy_total,top2_accuracy_total,top3_accuracy_total))
        stop = time.time()
        duration = stop-start
        # print("Time to validate: {0}".format(duration))

        if _WRITE==1:
          train_summary = tf.Summary()
          train_summary.value.add(tag="top1_accuracy", simple_value=top1_accuracy_total)
          train_summary.value.add(tag="top2_accuracy", simple_value=top2_accuracy_total)
          train_summary.value.add(tag="loss", simple_value=loss_total)
          writer_train_total.add_summary(train_summary, current_epoch)
          # with open(result_file, 'a') as f:
          #   f.write("Training set Check: The validation result for top 1 is {}%, for top 2 is {}% and for top 3 is {}% for step {}. The current epoch is {}.\n".format(top1_result*100,top2_result*100,top3_result*100, step,current_epoch))
        if int(best_accuracies['train'])< top1_accuracy_total:
          best_accuracies['train'] = top1_accuracy_total
          best_accuracies['train_epoch'] = current_epoch
        train_loss = np.append(train_loss,loss_total)  

      ########## Validation 1 ########

        print('Checking the Validation set..')
        tester=1
        top1_result_vector_val = []
        top2_result_vector_val = []
        top1_accuracy_total = 0
        top2_accuracy_total = 0
        start = time.time()
        loss_all = []


        while tester == 1:
          batch_xs, batch_ys, file_num_val, val_completed = next_feature_batch(batch_size, file_num_val, dataset_split['val'],dataset=dataset, frame_dim = frame_dim)
          # print('value of file_num_val is {}'.format(file_num_val))
          if val_completed ==1:
            tester=0
            file_num_val =0
            break
          [top1_result,top2_result,loss] = sess.run([top1_accuracy,top2_accuracy,loss_ret],feed_dict={model_input:batch_xs, y: batch_ys})
          if top1_result_vector_val == []:
            top1_result_vector_val = top1_result
            top2_result_vector_val = top2_result
            loss_all = loss 
          else:
            top1_result_vector_val=np.hstack((top1_result_vector_val,top1_result))
            top2_result_vector_val=np.hstack((top2_result_vector_val,top2_result))
            loss_all = np.append(loss_all,loss)
            # tester=0
        top1_accuracy_total = np.mean(top1_result_vector_val)
        top2_accuracy_total = np.mean(top2_result_vector_val)
        loss_total = np.mean(loss_all)
        # print("Unseen Actor Validation: The total accuracy for top 1 is {}, for top 2 is {}".format(top1_accuracy_total,top2_accuracy_total))
        # print('{} files validated.'.format(file_num_val))
        stop = time.time()
        duration = stop-start
        # print("Time to validate: {0}".format(duration))

        if _WRITE==1:
          val_summary = tf.Summary()
          val_summary.value.add(tag="top1_accuracy", simple_value=top1_accuracy_total)
          val_summary.value.add(tag="top2_accuracy", simple_value=top2_accuracy_total)
          val_summary.value.add(tag="loss", simple_value=loss_total)
          # val_summaries = sess.run(val_summaries, feed_dict={valid_accuracy1_placeholder: top1_accuracy_total, valid_accuracy2_placeholder: top2_accuracy_total, valid_accuracy2_placeholder: top2_accuracy_total})
          writer_val_total.add_summary(val_summary, current_epoch)
          # with open(result_file, 'a') as f:
        if int(best_accuracies['val'])< top1_accuracy_total:
          best_accuracies['val'] = top1_accuracy_total
          best_accuracies['val_epoch'] = current_epoch
        print('Validation completed. Epoch saved.')

      if len(train_loss)>10 and np.linalg.norm(train_loss[-6:-1]- train_loss[-5:])<0.001:
      	with open(result_file,'a') as f:
          f.write('Validation accuracy stabalized at epoch {}'.format(epoch))
    with open(result_file, 'a') as f:
      f.write('Best training accuracy: {} at epoch: {}.\n'.format(best_accuracies['train'],best_accuracies['train_epoch']))
      f.write('Best validation accuracy: {} at epoch: {}.\n'.format(best_accuracies['val'],best_accuracies['val_epoch']))
    # with open('out/'+split+'_accuracies.pkl',"wb") as f2:
    acc = ([int(best_accuracies['train']),int(best_accuracies['val'])])
    np.save('../results/out/{}/accuracies_{}.npy'.format(classifier,split),acc)

if __name__ == '__main__':
  tf.app.run(main)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf
import random
import os,sys
import time


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_dataset_info(name):
  return {
    'multiview_action_videos':{
      'num_classes': 10,
      'action_labels': {'a01':0,'a02':1,'a03':2,'a04':3,'a05':4,'a06':5,'a08':6,'a09':7,'a11':8,'a12':9}
    },
    'ixmas':{
      'num_classes': 11,
      'action_labels': {'checkwatch':0,'crossarms':1,'scratchhead':2,'sitdown':3,'getup':4,'turnaround':5,
      'walk':6,'wave':7,'punch':8,'kick':9,'pickup':10}
    },  
  }[name]

_VIDEO_FRAMES = 15
# dataset_root = 'mixed_5b_features/'

  
def next_batch(batch_size,file_num,filename,dataset = 'ixmas'):
  global _VIDEO_FRAMES
  dataset_root = '../dataset/{}/mixed_5b_features/'.format(dataset)
  action_labels = get_dataset_info(dataset)['action_labels']
  epoch_completed = False
  # prinfile_num=5t(file_num)
  file_temp = file_num
  batch = np.ones([2,2])
  y = np.zeros(batch_size)
  with open(filename) as f:
    content = f.readlines()
  files = [file.rstrip('\n') for file in content]
  # print(len(files))
  i=0
  for file in files:
    i+=1
    if i>=len(files): 
      # i=1
      # file_temp=1
      epoch_completed = True
      
      return 0,0,0,epoch_completed
    if i<file_temp: continue
    sample = np.load(os.path.join(dataset_root,file))
    frames = sample.shape[1]
    if frames< _VIDEO_FRAMES: continue
    # print(file)
    y[batch.shape[0]]= action_labels[file.split('/')[0]]
    extra = frames- _VIDEO_FRAMES
    start = random.randint(0,extra)
    if batch.shape == (2,2):
      batch = sample[:,start:start+_VIDEO_FRAMES,:,:]
      if batch_size ==1:
      	return batch,y,file_num,epoch_completed
      continue
    else: 
      # print(batch.shape)
      batch = np.concatenate((batch,sample[:,start:start+_VIDEO_FRAMES,:,:]),axis=0)
      if batch_size ==2:
      	return batch,y,file_num,epoch_completed
    if batch.shape[0]==batch_size: 
      file_num = i
      return batch,y,file_num,epoch_completed


def next_feature_batch(batch_size,file_num,filename,dataset = 'IXMAS',frame_dim = 5):
  global _VIDEO_FRAMES
  dataset_root = '../dataset/{}/mixed_5b_features/'.format(dataset)
  epoch_completed = False
  action_labels = get_dataset_info(str(dataset))['action_labels']
  # prinfile_num=5t(file_num)
  file_temp = file_num
  batch = np.ones([2,2])
  y = np.zeros(batch_size)
  with open(filename) as f:
    content = f.readlines()
  files = [file.rstrip('\n') for file in content]
  # print(len(files))
  i=0
  for file in files:
    i+=1
    if i>=len(files): 
      epoch_completed = True
      return None,None,0,epoch_completed
    if i<file_temp: continue # Skips all the files already used in this epoch.
    sample = np.load(os.path.join(dataset_root,file))
    frames = sample.shape[1]
    # print('frames: {}'.format(frames))
    # print(sample[:,0:min(frames,frame_dim-frames),:,:,:].shape)
    while frames< frame_dim: 
      frames = sample.shape[1]
      sample = np.concatenate((sample,sample[:,0:min(frames,frame_dim-frames),:,:,:]),axis=1)
    # print(file)
    # y[batch.shape[0]]= action_labels[file.split('/')[0]]
    extra = frames- frame_dim
    start = np.random.randint(0,high=extra+1)
    if batch.shape == (2,2):
      batch = sample[:,start:start+frame_dim,:,:]
      y[0] = action_labels[file.split('/')[0]]
      if batch_size ==1:
        return batch,y,file_num,epoch_completed
      continue
    else: 
      # print(batch.shape)
      y[batch.shape[0]]= action_labels[file.split('/')[0]]
      batch = np.concatenate((batch,sample[:,start:start+frame_dim:,:]),axis=0)
      if batch_size ==2:
        return batch,y,file_num,epoch_completed
    if batch.shape[0]==batch_size: 
      file_num = i
      return batch,y,file_num,epoch_completed
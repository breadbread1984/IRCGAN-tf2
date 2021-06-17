#!/usr/bin/python3

import h5py;
import numpy as np;
import tensorflow as tf;

def load_mnist_caption(filename):

  f = h5py.File(filename, 'r');
  data_train = np.array(f['mnist_gif_train']);
  captions_train = np.array(f['mnist_captions_train']);
  data_val = np.array(f['mnist_gif_val']);
  captions_val = np.array(f['mnist_captions_val']);
  f.close();
  train_data = tf.data.Dataset.from_tensor_slices(data_train);
  train_label = tf.data.Dataset.from_tensor_slices(captions_train);
  val_data = tf.data.Dataset.from_tensor_slices(data_val);
  val_label = tf.data.Dataset.from_tensor_slices(captions_val);
  return (train_data, train_label), (val_data, val_label);
  
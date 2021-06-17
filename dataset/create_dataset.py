#!/usr/bin/python3

import h5py;
import numpy as np;
import tensorflow as tf;

def create_dataset(filename):

  f = h5py.File(filename, 'r');
  data_train = np.array(f['mnist_gif_train']);
  captions_train = np.array(f['mnist_captions_train']);
  data_val = np.array(f['mnist_gif_val']);
  captions_val = np.array(f['mnist_captions_val']);
  f.close();
  write_tfrecord('trainset.tfrecord', data_train, captions_train);
  write_tfrecord('testset.tfrecord', data_val, captions_val);

def write_tfrecord(filename, data, caption):
  writer = tf.io.TFRecordWriter(filename);
  for i in range(data.shape[0]):
    sample1 = data[i];
    caption1 = caption[i];
    for j in range(data_train.shape[0]):
      if j == i:
        # write a sample with corresponding sample and caption
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'video': tf.train.Feature(float_list = tf.train.FloatList(value = sample1.reshape((-1,)))),
            'caption': tf.train.Feature(int64_list = tf.train.Int64List(value = caption1)),
            'matched': tf.train.Feature(int64_list = [1]),
          }
        ));
      elif caption1 != caption2:
        # write a sample with sample1 and caption2
        sample2 = data[j];
        caption2 = caption[j];
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'video': tf.train.Feature(float_list = tf.train.FloatList(value = sample1.reshape((-1,)))),
            'caption': tf.train.Feature(int64_list = tf.train.Int64List(value = caption2)),
            'matched': tf.train.Feature(int64_list = [0]),
          }
        ));
      writer.write(trainsample.SerializeToString());
  writer.close();

if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)' % argv[0]);
    exit(1);
  

#!/usr/bin/python3

import h5py;
import numpy as np;
import tensorflow as tf;

class SampleGenerator(object):  
  def __init__(self, filename):
    f = h5py.File(filename, 'r');
    self.data_train = np.array(f['mnist_gif_train']);
    self.captions_train = np.array(f['mnist_captions_train']);
    self.data_val = np.array(f['mnist_gif_val']);
    self.captions_val = np.array(f['mnist_captions_val']);
    f.close();
  def sample_generator(self, is_trainset = True):
    data = self.data_train if is_trainset else self.data_val;
    caption = self.captions_train if is_trainset else self.captions_val;
    def gen():
      for i in range(data.shape[0]):
        sample1 = np.transpose(data[i], (0,2,3,1));
        caption1 = caption[i];
        samples = np.arange(data.shape[0]);
        np.delete(samples, i);
        samples = np.random.choice(samples, 3, replace=False);
        for j in samples:
          sample2 = np.transpose(data[j], (0,2,3,1));
          caption2 = np.expand_dims(caption[j], axis = -1);      
          if np.all(caption1 == caption2):
            # write a sample with corresponding sample and caption
            yield sample1, caption2, 1;
          elif np.any(caption1 != caption2):
            # write a sample with sample1 and caption2
            yield sample1, caption2, 0;
          else:
            raise Exception('mustn\'t be here');
    return gen;
  def parse_function(self, sample, caption, label):
    sample = sample / 128. - 1; # in range [-1, 1]
    return sample, caption, label;
  def get_trainset(self,):
    return tf.data.Dataset.from_generator(self.sample_generator(True), (tf.float32, tf.int64, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,1,]), tf.TensorShape([]))).map(self.parse_function).repeat(-1);
  def get_testset(self):
    return tf.data.Dataset.from_generator(self.sample_generator(False), (tf.float32, tf.int64, tf.int64), (tf.TensorShape([16,64,64,1]), tf.TensorShape([9,1,]), tf.TensorShape([]))).map(self.parse_function).repeat(-1);

if __name__ == "__main__":

  import cv2;
  generator = SampleGenerator('mnist_single_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();
  cv2.namedWindow('sample');
  for sample, caption, matched in trainset:
    print(caption, matched);
    sample = (sample + 1) * 128.;
    for image in sample:
      cv2.imshow('sample',image.numpy().astype(np.uint8));
      cv2.waitKey(50);
  generator = SampleGenerator('mnist_two_gif.h5');
  trainset = generator.get_trainset();
  testset = generator.get_testset();

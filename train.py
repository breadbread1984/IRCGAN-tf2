#!/usr/bin/python3

import tensorflow as tf;
from models import TextEncoder, VideoGenerator, IntrospectiveDiscriminator;
from dataset.load_dataset import load_mnist_caption;

def main(filename = None, step = 10000):
  
  (train_data, train_label), (val_data, val_label) = load_mnist_caption(filename);
  train_data_iter = iter(train_data);
  train_label_iter = iter(train_label);
  val_data_iter = iter(val_data);
  val_label_iter = iter(val_label);
  for i in range(step):
    train_sample = next(train_data_iter);
    train_target = next(train_label_iter);
    val_sample = next(val_data_iter);
    val_target = next(val_label_iter);


if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s <dataset>' % argv[0]);
    exit(1);
  assert argv[1] in ['single','double'];
  main('mnist_single_gif.h5' if argv[1] == 'single' else 'mnist_two_gif.h5');

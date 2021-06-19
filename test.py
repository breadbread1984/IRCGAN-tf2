#!/usr/bin/python3

import numpy as np;
import cv2;
import tensorflow as tf;
import dataset.mnist_caption_single as single;
import dataset.mnist_caption_two_digit as double;
from models import VideoGenerator;

def main(digits, movings):

  assert len(digits) in [1,2];
  assert len(movings) in [1,2];
  assert np.all([digit in [*"0123456789"] for digit in digits]);
  assert np.all([moving in ['left and right','up and down'] for moving in movings]);
  assert len(digits) == len(movings);
  if len(digits) == 1:
    sentence = 'the digits %d is moving %s .' % (digits[0],movings[0]);
    tokens = sent2matrix(sentence, single.dictionary);
  else:
    sentence = 'digit %d is %s and digit %d is %s .' % (digits[0], movings[0], digits[1], movings[1]);
    tokens = sent2matrix(sentence, double.dictionary);
  print(sentence);
  e = tf.keras.models.load_model('encoder.h5', custom_objects = {'tf': tf});
  g = VideoGenerator();
  g.load_weights('generator_weights.h5');
  tokens = tf.expand_dims(tokens, axis = 0);
  code = e(tokens);
  videos = g(code);
  video = videos[0].numpy(); # video.shape = (16,64,64,1)
  writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (64,64), True);
  for frame in video:
    frame.astype(np.uint8);
    writer.write(frame);
  writer.release();

def input_digit(message):
  while True:
    digit = input(message);
    if digit not in [*'0123456789']:
      print('wrong digit, enter again');
      continue;
    break;
  return digit;

def input_movement(message):
  while True:
    choice = input('choose a movement from 0: \'left and right\', 1: \'up and down\'');
    if choice not in [*'01']:
      print('wrong movement choice, enter again');
      continue;
    break;
  return 'left and right' if choice == '0' else 'up and down';

if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)' % argv[0]);
    exit(1);
  if argv[1] == 'single':
    digit = input_digit('choose a digit from [0123456789]');
    movement = input_movement('choose a movement from 0: \'left and right\', 1: \'up and down\'');
    digits = [digit];
    movements = [movement];
  else:
    first_digit = input_digit('choose the first single digit from [0123456789]');
    first_movement = input_movement('choose a movement for the first digit from 0: \'left and right\', 1: \'up and down\'');
    second_digit = input_digit('choose the second single digit from [0123456789]');
    second_movement = input_movement('choose a movement for the second digit from 0: \'left and right\', 1: \'up and down\'');
    digits = [first_digit, second_digit];
    movements = [first_movement, second_movement];
  main(digits, movements);


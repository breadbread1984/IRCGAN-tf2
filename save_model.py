#!/usr/bin/python3

import tensorflow as tf;
from models import TextEncoder, VideoGenerator, IntrospectiveDiscriminator;

encoder_dim = 16;

def main(vocab_size):
  e = TextEncoder(vocab_size, encoder_dim);
  g = VideoGenerator();
  d = IntrospectiveDiscriminator();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  checkpoint = tf.train.Checkpoint(encoder = e, generator = g, discriminator = d, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  g.save('generator.h5');

if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)');
    exit(1);
  assert argv[1] in ['single', 'double'];
  if argv[1] == 'single':
    from dataset.mnist_caption_single import dictionary;
    vocab_size = len(dictionary);
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    vocab_size = len(dictionary);
  main(vocab_size);

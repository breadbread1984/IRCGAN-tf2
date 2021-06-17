#!/usr/bin/python3

import tensorflow as tf;
from models import TextEncoder, VideoGenerator, IntrospectiveDiscriminator;
from dataset.load_dataset import load_mnist_caption;

batch_size = 4;
encoder_dim = 16;

def main(filename = None, vocab_size = None, step = 10000, val_interval = 100):
  
  (train_data, train_label), (val_data, val_label) = load_mnist_caption(filename);
  train_data_iter = iter(train_data.batch(batch_size));
  train_label_iter = iter(train_label.batch(batch_size));
  val_data_iter = iter(val_data.batch(batch_size));
  val_label_iter = iter(val_label.batch(batch_size));
  encoder = TextEncoder(vocab_size, encoder_dim);
  g = VideoGenerator();
  d = IntrospectiveDiscriminator();
  for i in range(step):
    real = tf.transpose(next(train_data_iter),(0,1,3,4,2)); # train_sample = (batch, 16, 64, 64, 1)
    caption = next(train_label_iter); # train_caption.shape = (batch, 9)
    code = encoder(caption); # code.shape = (batch, 16)
    fake = g(code); # fake.shape = (batch, 16, 64, 64, 1)
    real_motion_disc, real_frame_disc, real_text_disc, real_recon_latent0, real_recon_latent1 = d([real, code]);
    fake_motion_disc, fake_frame_disc, fake_text_disc, fake_recon_latent0, fake_recon_latent1 = d([fake, code]);
    if i % val_interval == 0:
      real = next(val_data_iter);
      caption = next(val_label_iter);
      # TODO
    

if __name__ == "__main__":

  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s (single|double)' % argv[0]);
    exit(1);
  assert argv[1] in ['single','double'];
  if argv[1] == 'single':
    from dataset.mnist_caption_single import dictionary;
    vocab_size = len(dictionary);
  elif argv[1] == 'double':
    from dataset.mnist_caption_two_digit import dictionary;
    vocab_size = len(dictionary);
  main('mnist_single_gif.h5' if argv[1] == 'single' else 'mnist_two_gif.h5', vocab_size);


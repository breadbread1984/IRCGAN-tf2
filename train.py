#!/usr/bin/python3

from os import mkdir;
from os.path import exists;
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
  e = TextEncoder(vocab_size, encoder_dim);
  g = VideoGenerator();
  d = IntrospectiveDiscriminator();
  real_labels = tf.ones((batch_size,));
  fake_labels = tf.zeros((batch_size,));
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedulers.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  if False == exists('checkpoint'): mkdir('checkpoint');
  checkpoint = tf.train.Checkpoint(encoder = e, generator = g, discriminator = d, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoint'));
  for i in range(step):
    real = tf.transpose(next(train_data_iter),(0,1,3,4,2)); # train_sample = (batch, 16, 64, 64, 1)
    caption = next(train_label_iter); # train_caption.shape = (batch, 9)
    with tf.GradientTape(persistent = True) as tape:
      code = e(caption); # code.shape = (batch, 16)
      fake = g(code); # fake.shape = (batch, 16, 64, 64, 1)
      real_motion_disc, real_frame_disc, real_text_disc, real_recon_latent0, real_recon_latent1 = d([real, code]);
      fake_motion_disc, fake_frame_disc, fake_text_disc, fake_recon_latent0, fake_recon_latent1 = d([fake, code]);
      d_real_loss = tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, real_motion_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, real_frame_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, real_text_disc);
      d_fake_loss = tf.keras.losses.SparseCategoricalCrossentropy()(fake_labels, fake_motion_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(fake_labels, fake_frame_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(fake_labels, fake_text_disc);
      g_loss = tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, fake_motion_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, fake_frame_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(real_labels, fake_text_disc);
      

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


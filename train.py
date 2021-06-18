#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
import tensorflow as tf;
from models import TextEncoder, VideoGenerator, IntrospectiveDiscriminator;
from dataset.sample_generator import SampleGenerator;

batch_size = 4;
encoder_dim = 16;

def main(filename = None, vocab_size = None, val_interval = 100):
  
  dataset_generator = SampleGenerator(filename);
  trainset = dataset_generator.get_trainset().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = dataset_generator.get_testset().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  trainset_iter = iter(trainset);
  testset_iter = iter(testset);
  e = TextEncoder(vocab_size, encoder_dim);
  g = VideoGenerator();
  d = IntrospectiveDiscriminator();
  true_labels = tf.ones((batch_size,));
  false_labels = tf.zeros((batch_size,));
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedulers.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(encoder = e, generator = g, discriminator = d, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_disc_loss = tf.keras.metrics.Mean(name = 'discriminator loss', dtype = tf.float32);
  avg_gen_loss = tf.keras.metrics.Mean(name = 'generator loss', dtype = tf.float32);
  while True:
    real, caption, matched = next(trainset_iter);
    with tf.GradientTape(persistent = True) as tape:
      code = e(caption); # code.shape = (batch, 16)
      fake = g(code); # fake.shape = (batch, 16, 64, 64, 1)
      real_motion_disc, real_frame_disc, real_text_disc, real_recon_latent0, real_recon_latent1 = d([real, code]);
      fake_motion_disc, fake_frame_disc, fake_text_disc, fake_recon_latent0, fake_recon_latent1 = d([fake, code]);
      # NOTE: D1: text_disc, D2: motion_disc & frame_disc
      # NOTE: Q1: real_recon_latent1, Q2: real_recon_latent0
      # 1.1) matched discriminator loss
      d1_loss = tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, real_text_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, fake_text_disc);
      d2_loss = tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, real_motion_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, real_frame_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, fake_motion_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, fake_frame_disc);
      q1_loss = tf.keras.losses.MeanSquaredError()(code, real_recon_latent1);
      q2_loss = tf.keras.losses.MeanSquaredError()(code, real_recon_latent0);
      matched_disc_loss = d1_loss + d2_loss + q1_loss + q2_loss;
      # 1.2) unmatched discriminator loss
      d1_loss = tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, real_text_disc);
      d2_loss = tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, real_motion_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, real_frame_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, fake_motion_disc) \
                + tf.keras.losses.SparseCategoricalCrossentropy()(false_labels, fake_frame_disc);
      unmatched_disc_loss = d1_loss + d2_loss;
      # 1.3) discriminator loss
      disc_loss = tf.where(tf.math.equal(matched,1),matched_disc_loss,unmatched_disc_loss);
      # 2) generator loss
      g1_loss = tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, fake_text_disc);
      g2_loss = tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, fake_motion_disc) \
              + tf.keras.losses.SparseCategoricalCrossentropy()(true_labels, fake_frame_disc);
      info_loss = tf.keras.losses.MeanSquaredError()(code, fake_recon_latent1) \
                + tf.keras.losses.MeanSquaredError()(code, fake_recon_latent0);
      gen_loss = g1_loss + g2_loss + info_loss;
    avg_disc_loss.update_state(disc_loss);
    avg_gen_loss.update_state(gen_loss);
    # 3) gradients
    d_grads = tape.gradient(disc_loss, d.trainable_variables);
    g_grads = tape.gradient(gen_loss, g.trainable_variables);
    e_grads = tape.gradient(disc_loss + gen_loss, e.trainable_variables);
    optimizer.apply_gradients(zip(d_grads, d.trainable_variables));
    optimizer.apply_gradients(zip(g_grads, g.trainable_variables));
    optimizer.apply_gradients(zip(e_grads, e.trainable_variables));
    if tf.equal(optimizer.iterations % val_interval, 0):
      checkpoint.save(join('checkpoint','ckpt'));
      with log.as_default():
        tf.summary.scalar('discriminator loss', avg_disc_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('generator loss', avg_gen_loss.result(), step = optimizer.iterations);
      avg_disc_loss.reset_states();
      avg_gen_loss.reset_states();

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

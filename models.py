#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def TextEncoder(src_vocab_size, input_dims, units = 16):
  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  results = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(inputs); # results.shape = (batch, ragged length)
  results = tf.keras.layers.Embedding(src_vocab_size, input_dims)(results); # results.shape = (batch, ragged length, input_dims)
  results = tf.keras.layers.Bidirectional(
    layer = tf.keras.layers.LSTM(units // 2),
    backward_layer = tf.keras.layers.LSTM(units // 2, go_backwards = True),
    merge_mode = 'concat')(results); # results.shape = (batch, units)
  return tf.keras.Model(inputs = inputs, outputs = results);

def RecurrentTransconvolutionalGenerator(channels = 16, layers = 5, img_channels = 3):
  inputs = tf.keras.Input((channels,)); # inputs.shape = (batch, channels)
  hiddens = [tf.keras.Input((2**i * 2**i,)) for i in range(layers)]; # hiddens[i].shape = (batch * channels * 2, 2^i * 2^i)
  cells = [tf.keras.Input((2**i * 2**i,)) for i in range(layers)]; # cells[i].shape = (batch * channels * 2, 2^i * 2^i)
  results = tf.keras.layers.Reshape((1, 1, channels))(inputs); # results.shape = (batch, 1, 1, channels)
  dnorm = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape = tf.shape(x)))(results); # dnorm.shape = (batch, 1, 1, channel)
  results = tf.keras.layers.Concatenate(axis = -1)([results, dnorm]); # results.shape = (batch, 1, 1, 2 * channels)
  next_hiddens = list();
  next_cells = list();
  for i in range(layers):
    before = results; # before.shape = (batch, 2^i, 2^i, 2 * channels)
    results = tf.keras.layers.Lambda(lambda x, l: tf.reshape(tf.transpose(x, (0,3,1,2)), (-1, 2**l * 2**l)), arguments = {'l': i})(results); # results.shape = (batch * 2*channels, 1, 2^i, 2^i)
    lstm_inputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1))(results); # lstm.shape = (batch, 1, 2^i * 2^i)
    results, hidden, cell = tf.keras.layers.LSTM(units = 2**i * 2**i, return_state = True)(lstm_inputs, initial_state = (hiddens[i], cells[i])); # hidden.shape = (batch * 2 * channels, 2^i * 2^i)
    next_hiddens.append(hidden);
    next_cells.append(cell);
    after = tf.keras.layers.Lambda(lambda x, c, l: tf.transpose(tf.reshape(x, (-1, 2 * c, 2**l, 2**l)), (0, 2, 3, 1)), arguments = {'l': i, 'c': channels})(results);  # results.shape = (batch, 2^i, 2^i, 2 * channels)
    results = tf.keras.layers.Concatenate(axis = -1)([before, after]); # results.shape = (batch, 2^i, 2^i, 2 * channels)
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * channels, kernel_size = (3, 3), strides = (2,2), padding = 'same')(results); # results.shape = (batch, 2^(i+1), 2^(i+1), 2 * channels)
    results = tf.keras.layers.BatchNormalization()(results); # results.shape = (batch, 2^(i+1), 2^(i+1), 2 * channels)
    results = tf.keras.layers.LeakyReLU()(results); # results.shape = (batch, 2^(i+1), 2^(i+1), 2 * channels)
  results = tf.keras.layers.Conv2DTranspose(filters = img_channels, kernel_size = (3, 3), strides = (2,2), padding = 'same')(results); # results.shape = (batch, 64, 64, img_channels)
  return tf.keras.Model(inputs = (inputs, *hiddens, *cells), outputs = (results, *next_hiddens, *next_cells));

class VideoGenerator(tf.keras.Model):
  def __init__(self, filters = 16, layers = 5, img_channels = 1, length = 16):
    super(VideoGenerator, self).__init__();
    self.generator = RecurrentTransconvolutionalGenerator(channels = filters, layers = layers, img_channels = img_channels);
    self.filters = filters;
    self.layer_num = layers;
    self.length = length;
  def call(self, inputs):
    hiddens = [tf.zeros((inputs.shape[0] * 2 * self.filters, 2**i * 2**i)) for i in range(self.layer_num)];
    cells = [tf.zeros((inputs.shape[0] * 2 * self.filters, 2**i * 2**i)) for i in range(self.layer_num)];
    video = list();
    for i in range(self.length):
      outputs = self.generator([inputs, *hiddens, *cells]);
      frame = outputs[0]; # frame.shape = (batch, height = 64, width = 64, img_channels = 1)
      video.append(frame);
      hiddens = outputs[1:6];
      cells = outputs[6:11];
    video = tf.stack(video, axis = 1); # video.shape = (batch, length = 16, height = 64, width = 64, img_channels = 1)
    return video;

def IntrospectiveDiscriminator(img_size = 64, img_channels = 1, length = 16, units = 16):
  video = tf.keras.Input((length, img_size, img_size, img_channels)); # video.shape = (batch, length = 16, height = 64, width = 64, img_channls = 1)
  text = tf.keras.Input((units,)); # text.shape = (batch, units)
  # 1) temporal coherence
  results = tf.keras.layers.Lambda(lambda x, s, c: tf.reshape(x, (-1, s, s, c)), arguments = {'s': img_size, 'c': img_channels})(video); # results.shape = (batch * length, height, width, img_channels)
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same')(results); # results.shape = (batch * length, height / 2, width / 2, 64)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = 'same')(results); # results.shape = (batch * length, height / 4, width / 4, 128)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2), padding = 'same')(results); # results.shape = (batch * length, height / 8, width / 8, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2), padding = 'same')(results); # results.shape = (batch * length, height / 16, width / 16, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  motion_frame_inputs = tf.keras.layers.LeakyReLU()(results);
  # 1.1) motion loss
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same')(motion_frame_inputs); # results.shape = (batch * length, height / 16, width / 16, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch * length, height / 16, width / 16, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l, x.shape[-3], x.shape[-2], x.shape[-1])), arguments = {'l': length})(results); # results.shape = (batch, length, height / 16, width / 16, 256)
  results = tf.keras.layers.Lambda(lambda x: x[:,1:,...] - x[:,:-1,...])(results); # results.shape = (batch, length - 1, height / 16, width / 16, 256)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1])))(results); # results.shape = (batch * (length - 1), height / 16, width / 16, 256)
  motion_disc = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l-1, x.shape[-3], x.shape[-2], x.shape[-1])), arguments = {'l': length})(results); # motion_disc.shape = (batch, length - 1, height / 16, width / 16, 256)
  motion_disc = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(motion_disc); # motion_disc.shape = (batch, 256)
  motion_disc = tf.keras.layers.Dense(2, activation = tf.keras.activations.softmax)(motion_disc); # motion_disc.shape = (batch, 2)
  recon_latent0_left = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l-1, x.shape[-3], x.shape[-2], x.shape[-1])), arguments = {'l': length})(results); # recon_latent0.shape = (batch, length - 1, height / 16, width / 16, 256)
  recon_latent0_left = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(recon_latent0_left); # recon_latent0.shape = (batch, 256)
  # 1.2) frame loss
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same')(motion_frame_inputs); # results.shape = (batch * length, height / 16, width / 16, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch * length, height / 16, width / 16, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  frame_disc = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l, x.shape[-3], x.shape[-2], x.shape[-1])), arguments = {'l': length})(results); # frame_disc.shape = (batch, length, height / 16, width / 16, 256)
  frame_disc = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(frame_disc); # frame_disc.shape = (batch, 256)
  frame_disc = tf.keras.layers.Dense(2, activation = tf.keras.activations.softmax)(frame_disc); # frame_disc.shape = (batch, 2)
  recon_latent0_right = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l, x.shape[-3], x.shape[-2], x.shape[-1])), arguments = {'l': length})(results); # recon_latent1.shape = (batch, length, height / 16, width / 16, 256)
  recon_latent0_right = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(recon_latent0_right); # recon_latent1.shape = (batch, 256)
  recon_latent0 = tf.keras.layers.Concatenate(axis = -1)([recon_latent0_left, recon_latent0_right]); # recon_latent0.shape = (batch, 512)
  recon_latent0 = tf.keras.layers.Dense(units, activation = tf.keras.activations.tanh)(recon_latent0); # recon_latent1.shape = (batch, units)
  # 2) whole video
  results = tf.keras.layers.Conv3D(filters = 64, kernel_size = (3,3,3), strides = (2,2,2), padding = 'same')(video); # results.shape = (batch, length / 2, height / 2, width / 2, 64)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv3D(filters = 128, kernel_size = (3,3,3), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch, length / 4, height / 4, width / 4, 128)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch, length / 8, height / 8, width / 8, 256)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch, length / 16, height / 16, width / 16, 512)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  # 2.1) video loss
  video_results = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,3,3), strides = (1,2,2), padding = 'same')(results); # video_results.shape = (batch, length / 16, height / 32, width / 32, 512)
  video_results = tf.keras.layers.BatchNormalization()(video_results);
  video_results = tf.keras.layers.LeakyReLU()(video_results);
  video_results = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,3,3), strides = (1,2,2), padding = 'same')(video_results); # video_results.shape = (batch, length / 16, height / 64, width / 64, 512)
  recon_latent1 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(video_results); # video_results.shape = (batch, 512)
  recon_latent1 = tf.keras.layers.Dense(units, activation = tf.keras.activations.tanh)(recon_latent1); # recon_latent1.shape = (batch, units)
  # 2.2) text loss
  text_results = tf.keras.layers.Reshape((1, 1, units))(text); # text_results.shape = (batch, 1, 1, units)
  text_results = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = (2,2), padding = 'same')(text_results); # text_results.shape = (batch, 2, 2, 256)
  text_results = tf.keras.layers.BatchNormalization()(text_results);
  text_results = tf.keras.layers.LeakyReLU()(text_results);
  text_results = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = (2,2), padding = 'same')(text_results); # text_results.shape = (batch, 4, 4, 256)
  text_results = tf.keras.layers.BatchNormalization()(text_results);
  text_results = tf.keras.layers.LeakyReLU()(text_results);
  text_results = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1))(text_results); # text_results.shape = (batch, 1, 4, 4, 256)
  text_results = tf.keras.layers.Concatenate(axis = -1)([results, text_results]); # text_results.shape = (batch, 1, 4, 4, 768)
  text_results = tf.keras.layers.Conv3D(filters = 512, kernel_size = (1,1,1), strides = (1,1,1), padding = 'same')(text_results); # text_results.shape = (batch, 1, 4, 4, 512)
  text_results = tf.keras.layers.BatchNormalization()(text_results);
  text_results = tf.keras.layers.LeakyReLU()(text_results);
  text_disc = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, (1,2,3)))(text_results); # text_results.shape = (batch, 512)
  text_disc = tf.keras.layers.Dense(2, activation = tf.keras.activations.softmax)(text_disc); # text_disc.shape = (batch, 2)
  return tf.keras.Model(inputs = (video, text), outputs = (motion_disc, frame_disc, text_disc, recon_latent0, recon_latent1));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  encoder = TextEncoder(100,64);
  encoder.save('encoder.h5');
  inputs = np.random.randint(low = 0, high = 100, size = (50,));
  inputs = tf.RaggedTensor.from_row_lengths(inputs, [50,]);
  inputs = tf.expand_dims(inputs, axis = -1);
  results = encoder(inputs);
  print(results.shape);
  generator = RecurrentTransconvolutionalGenerator();
  generator.save('generator.h5');
  inputs = np.random.normal(size = (8, 16));
  hiddens = [np.random.normal(size = (8 * 2 * 16, 2**i * 2**i)) for i in range(5)];
  cells = [np.random.normal(size = (8 * 2 * 16, 2**i * 2**i)) for i in range(5)];
  results, hidden1, hidden2, hidden3, hidden4, hidden5, cell1, cell2, cell3, cell4, cell5 = generator([inputs, *hiddens, * cells]);
  print(results.shape);
  print(hidden1.shape);
  print(hidden2.shape);
  print(hidden3.shape);
  print(hidden4.shape);
  print(hidden5.shape);
  vgen = VideoGenerator();
  vgen.save_weights('vgen.h5');
  video = vgen(inputs);
  print(video.shape);
  disc = IntrospectiveDiscriminator();
  disc.save('disc.h5');
  motion_disc, frame_disc, text_disc, recon_latent0, recon_latent1 = disc([video, inputs]);
  print(motion_disc.shape);
  print(frame_disc.shape);
  print(text_disc.shape);
  print(recon_latent0.shape);
  print(recon_latent1.shape);

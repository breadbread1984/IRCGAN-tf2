#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def TextEncoder(src_vocab_size, input_dims, units = 8):
  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  results = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(inputs); # results.shape = (batch, ragged length)
  results = tf.keras.layers.Embedding(src_vocab_size, input_dims)(results); # results.shape = (batch, ragged length, input_dims)
  results = tf.keras.layers.Bidirectional(
    layer = tf.keras.layers.LSTM(units),
    backward_layer = tf.keras.layers.LSTM(units, go_backwards = True),
    merge_mode = 'concat')(results); # results.shape = (batch, 2 * encoder_params['units'])
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
  def __init__(self, channels = 16, layers = 5, img_channels = 1, length = 16):
    super(VideoGenerator, self).__init__();
    self.generator = RecurrentTransconvolutionalGenerator(channels = channels, layers = layers, img_channels = img_channels);
    self.channels = channels;
    self._layers = layers;
    self.length = length;
  def call(self, inputs):
    hiddens = [tf.zeros((inputs.shape[0] * 2 * self.channels, 2**i * 2**i) for i in range(self._layers))];
    cells = [tf.zeros((inputs.shape[0] * 2 * self.channels, 2**i * 2**i) for i in range(self._layers))];
    video = list();
    for i in range(self.length):
      outputs = self.generator([inputs, *hiddens, *cells]);
      frame = outputs[0]; # frame.shape = (batch, height = 64, width = 64, img_channels = 1)
      video.append(frame);
      hiddens = outputs[1:6];
      cells = outputs[6:11];
    video = tf.stack(video, axis = 1); # video.shape = (batch, length = 16, height = 64, width = 64, img_channels = 1)
    return video;

if __name__ == "__main__":

  encoder = TextEncoder(100,32);
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

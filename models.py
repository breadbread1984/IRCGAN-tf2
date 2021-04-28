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

def RecurrentTransconvolutionalGenerator(channels = 16, layers = 5):
  inputs = tf.keras.Input((channels,)); # inputs.shape = (batch, channels)
  initial_states = [(tf.keras.Input((2**i * 2**i * channels * 2,)), tf.keras.Input((2**i * 2**i * channels * 2,))) for i in range(layers)]; # init_states[i].shape = (batch, 2^i * 2^i * channels * 2)
  results = tf.keras.layers.Reshape((1, 1, channels))(inputs); # results.shape = (batch, 1, 1, channels)
  cells = list();
  for i in range(layers):
    dnorm = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape = tf.shape(x)))(results); # dnorm.shape = (batch, 2^i, 2^i, channel)
    results = tf.keras.layers.Concatenate(axis = -1)([results, dnorm]); # results.shape = (batch, 2^i, 2^i, 2 * channels)
    lstm_inputs = tf.keras.layers.Flatten()(results); # lstm_inputs.shape = (batch, 2^i * 2^i * channels * 2)
    hidden, cell = tf.keras.layers.LSTM(units = 2**i * 2**i * 2 * channels, return_state = True)(lstm_inputs, initial_state = initial_states[i]); # hidden.shape = (batch, 2^i * 2^i * channels * 2)
    cells.append(cell);
    results = tf.keras.layers.Reshape((2**i, 2**i, 2 * channels))(hidden); # results.shape = (batch, 2^i, 2^i, 2 * channels)
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * channels, kernel_size = (2, 2), padding = 'valid')(results); # results.shape = (batch, 2^(i+1), 2^(i+1), 2 * channels)
  return tf.keras.Model(inputs = (inputs, *initial_states), outputs = (results, *cells));

if __name__ == "__main__":

  encoder = TextEncoder(100,32);
  inputs = np.random.randint(low = 0, high = 100, size = (50,));
  inputs = tf.RaggedTensor.from_row_lengths(inputs, [50,]);
  inputs = tf.expand_dims(inputs, axis = -1);
  results = encoder(inputs);
  print(results.shape);
  generator = RecurrentTransconvolutionalGenerator();
  inputs = np.random.normal(size = (8, 16));
  initial_states = [(np.random.normal(size = (8, 2**i * 2**i * 32)), np.random.normal(size = (8, 2**i * 2**i * 32))) for i in range(5)];
  results, init1, init2, init3, init4, init5 = generate([inputs, * initial_states]);
  print(results.shape);
  print(init1.shape);
  print(init2.shape);
  print(init3.shape);
  print(init4.shape);
  print(init5.shape);

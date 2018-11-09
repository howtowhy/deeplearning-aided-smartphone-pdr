import math

import models
import tensorflow as tf
#import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return output #{"predictions": output}



class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, rnn_size=None, rnn_layers=None, rnn_type=None, multimodal_type=None, mm_pooling_dim=None, **unused_params):
    """Creates a model which uses a stack of LSTMs tmo represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    rnn_size = 16 #rnn_size or FLAGS.rnn_size
    rnn_layers = 1 #rnn_layers or FLAGS.rnn_layers

    rnn_cell = tf.contrib.rnn.BasicLSTMCell
    rnn_params = {"forget_bias":1.0}

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                rnn_cell(rnn_size, **rnn_params)
                for _ in range(rnn_layers)
                ])

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    logistic_model = LogisticModel()

    return logistic_model.create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

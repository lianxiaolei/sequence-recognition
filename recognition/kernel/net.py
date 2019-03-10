# coding:utf8

import tensorflow as tf


DIGITS = '0123456789'
# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 280, 28, 10, len(characters) + 1


class CRNN(object):
  """

  """
  def __init__(self, num_class, flags=None, tfrecord=None):
    self.num_class = num_class
    self.flags = flags
    if tfrecord:
      with tf.name_scope(name='ph'):
        self.X = tfrecord['input_batch']
        self.y = tfrecord['sparse_target_batch']
        self.seq_len = tfrecord['seqlen_batch']
        self.keep_prob = tf.placeholder(tf.float32, name='kp')

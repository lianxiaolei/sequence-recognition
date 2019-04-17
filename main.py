# coding:utf8

import tensorflow as tf
import sys
sys.path.append('/Users/imperatore/PycharmProjects/question-recognition/')
from recognition.kernel.network_tfrecord import run_multiprocess


if __name__ == '__main__':
  tf.app.flags.DEFINE_boolean("allow_soft_placement",
                              True, "Allow device soft device placement")
  tf.app.flags.DEFINE_boolean("log_device_placement",
                              False, "Log placement of ops on devices")
  tf.app.flags.DEFINE_integer("batch_size",
                              128, "Batch Size (default: 64)")
  tf.app.flags.DEFINE_float("dropout_keep_prob",
                            1.0, "Dropout keep probability")
  tf.app.flags.DEFINE_integer("evaluate_every",
                              100, "Evaluate model on dev set after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer('rnn_units',
                              128, "Rnn Units")
  # 初始化学习速率
  tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 1e-3, 'Learning rate initial value')
  tf.app.flags.DEFINE_integer('DECAY_STEPS', 10000, 'DECAY_STEPS')
  tf.app.flags.DEFINE_integer('REPORT_STEPS', 100, 'REPORT_STEPS')
  tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.0, 'LEARNING_RATE_DECAY_FACTOR')

  # crnn = CRNN(n_class)
  # crnn.architecture(input_shape=[None, width, height, 1])
  # print('Build model done!')
  # run_multiprocess('../../dataset/sequence_10k.tfrecord')
  run_multiprocess('/Users/imperatore/data/numbers_10k.tfrecord')
  print('Training done!')

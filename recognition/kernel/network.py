# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from abc import ABCMeta, abstractmethod
from tensorflow.contrib import rnn
import numpy as np
import time
import os
from recognition.kernel.sparse_parser import *
from recognition.kernel.data_provider import *
import sys

DIGITS = '0123456789'
# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 400, 80, 10, len(characters) + 1


class CRNN():
    """

    """

    def __init__(self, num_class, rnn_units):
        self.rnn_units = rnn_units
        self.num_class = num_class

    def _init_variable(self, shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def _change_size(self, input_shape, channel=None, auto_change_channel=True):
        """
        变换特征图尺寸，CNN中使用
        Args:
            :param input_shape:
            :param channel:
            :param auto_change_channel:
        :return: 改变后的尺寸
        """
        input_shape[2] = input_shape[2] // 2
        if auto_change_channel:
            input_shape[3] = input_shape[3] * 2
        if not channel:
            input_shape[3] = channel
        return input_shape

    def image2head(self, x):
        for i in range(3):
            x = tf.nn.conv2d(x, eval('self.w%s0' % i), [1, 2, 2, 2],
                             padding='same', name='cnn0%s' % i)
            tf.nn.relu(x)

            x = tf.nn.conv2d(x, eval('self.w%s1' % i), [1, 2, 2, 2],
                             padding='same', name='cnn1%s' % i)
            tf.nn.relu(x)

            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='valid', name='cnn2%s' % i)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        return x

    def head2tail(self, x):
        # x = slim.fully_connected(x, self.rnn_units)

        cell = rnn.LSTMCell(self.rnn_units, use_peepholes=True)
        back_cell = rnn.LSTMCell(self.rnn_units, back_x=True, use_peepholes=True)

        x, _ = tf.nn.dynamic_rnn(cell, x, self.seq_len)
        back_x, _ = tf.nn.dynamic_rnn(back_cell, x, self.seq_len)
        x = tf.add(x, back_x)

        batch_s, max_timesteps = tf.shape(self.X)

        x = tf.reshape(x, [-1, self.rnn_units])
        x = slim.fully_connected(x, self.num_class, activation_fn='softmax',
                                 weights_initializer='truncated_normal',
                                 biases_initializer='truncated_normal')
        # Reshape the feature map to (max_timesteps, batch_size, num_classes)
        x = tf.reshape(x, [batch_s, -1, self.num_class])
        # x = tf.transpose(x, (1, 0, 2))

        self.output = x
        return x

    def ctc_loss(self, x, label):
        return tf.nn.ctc_loss(label, x, sequence_length=self.seq_len)

    def architecture(self, input_shape, lr=1e-2, epoch=1e1, mode='train'):
        self.FLAGS = tf.flags.FLAGS

        config = tf.ConfigProto(
            allow_soft_placement=self.FLAGS.allow_soft_placement,  # 设置让程序自动选择设备运行
            log_device_placement=self.FLAGS.log_device_placement)

        self.sess = tf.Session(config=config)

        with tf.Graph().as_default():
            with tf.name_scope(name='ph'):
                self.X = tf.placeholder(shape=input_shape, name='x')
                self.y = tf.sparse_placeholder(tf.int32, name='y')  # 不指定shape时，可以feed任意shape
                self.seq_len = tf.sparse_placeholder(tf.int32, [None], name='seq_len')

                self.keep_prob = tf.placeholder(tf.float32, name='kp')

            with tf.name_scope(name='cnn_kernels'):
                self.w00 = self._init_variable(self._change_size(input_shape, channel=32,
                                                                 auto_change_channel=False),
                                               name='conv_w00')
                self.w01 = self._init_variable(self.w00.shape, name='conv_w01')

                self.w10 = self._init_variable(self._change_size(input_shape), name='conv_w10')
                self.w11 = self._init_variable(self.w10.shape, name='conv_w11')

                self.w20 = self._init_variable(self._change_size(input_shape), name='conv_w20')
                self.w21 = self._init_variable(self._change_size(input_shape), name='conv_w21')

            with tf.name_scope('architecture'):
                self.head = self.image2head(self.X)
                self.tail = self.head2tail(self.head)

            with tf.name_scope('loss'):
                self.loss = self.ctc_loss(self.y, self.tail)

            with tf.name_scope('accuracy'):
                self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.output,
                                                                 self.seq_len,
                                                                 merge_repeated=False)

                batch_acc = 0.
                decout = self.decoded.eval()
                if decout.shape[1] == self.seq_len:
                    batch_acc += (self.y == decout).all(axis=1).mean()
                self.acc = batch_acc

            with self.sess.as_default():
                self.learning_rate = 1e-3

                self.global_step = tf.Variable(0, name='global_step', trainable=True)

                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

                self.sess.run(tf.global_variables_initializer())

    def calc_accuracy(self, decode_list, test_target):
        original_list = decode_sparse_tensor(test_target)
        detected_list = decode_sparse_tensor(decode_list)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        self.accuracy = true_numer / len(original_list)
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def _accuracy(self):
        test_inputs, test_targets, test_seq_len = get_next_batch(self.FLAGS.batch_size)
        test_feed = {self.X: test_inputs,
                     self.y: test_targets,
                     self.seq_len: test_seq_len}
        dd, log_probs, accuracy = self.sess.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_targets)

    def summary(self):
        # Summary
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run_temp', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # Summary for loss and accuracy
        loss_summary = tf.summary.scalar('loss', self.loss)
        acc_summary = tf.summary.scalar('accuracy', self.accuracy)

    def train_step(self, x_batch, y_batch):
        y = sparse_tuple_from(y_batch)
        feed_dict = {
            self.X: x_batch,
            self.y: y,
            self.keep_prob: self.FLAGS.dropout_keep_prob
        }

        _, step, summaries, loss, accuracy = self.sess.run(
            [self.train_op, self.global_step,
             ]
        )

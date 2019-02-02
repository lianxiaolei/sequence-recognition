# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import numpy as np
import datetime
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

    def __init__(self, num_class):
        self.num_class = num_class

    def _init_variable(self, shape, name=None):
        # print('Define variable {} of shape {}'.format(name, shape))
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
            # print('Defind layer use kernel {} with name {}'.format('self.w%s0' % i, 'cnn0%s' % i))
            # print(eval('self.w%s0' % i))
            x = tf.nn.conv2d(x, eval('self.w%s0' % i), [1, 1, 1, 1],
                             padding='SAME', name='cnn0%s' % i)

            tf.nn.relu(x)

            x = tf.nn.conv2d(x, eval('self.w%s1' % i), [1, 1, 1, 1],
                             padding='SAME', name='cnn1%s' % i)
            tf.nn.relu(x)

            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='cnn2%s' % i)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        return x

    def head2tail(self, x):
        self.rnn_units = self.FLAGS.rnn_units
        # x = slim.fully_connected(x, self.rnn_units)
        # x = tf.squeeze(x, axis=-1)
        # time major 模式需要的input shape:(max_time x batch_size x num_classes)
        x = tf.transpose(x, (1, 0, 2, 3))

        shape = x.get_shape().as_list()
        x = tf.reshape(x, shape=[shape[0], -1, shape[2] * shape[3]])
        x = slim.fully_connected(x, self.rnn_units)

        cell = rnn.LSTMCell(self.rnn_units, use_peepholes=True, name='frnn')
        back_cell = rnn.LSTMCell(self.rnn_units, use_peepholes=True, name='brnn')

        initial_state_fw = cell.zero_state(shape[0], dtype=tf.float32)
        initial_state_bw = back_cell.zero_state(shape[0], dtype=tf.float32)

        # 构建双向叠加RNN
        print('双向RNN的输入(time_major)', x)
        print('双向RNN的参数(time_major)', initial_state_fw)
        x, _ = tf.nn.bidirectional_dynamic_rnn(cell, back_cell, x, self.seq_len,
                                               # initial_state_fw,
                                               # initial_state_bw,
                                               dtype=tf.float32, time_major=True)
        x = tf.add(x[0], x[1], name='add')

        max_timesteps, batch_s, _ = x.get_shape().as_list()

        # x = tf.reshape(x, [-1, self.rnn_units])
        x = slim.fully_connected(x, self.num_class, activation_fn=tf.nn.softmax)
        # Reshape the feature map to (max_timesteps, batch_size, num_classes)
        # x = tf.reshape(x, [-1, max_timesteps, self.num_class])

        return x

    def ctc_loss(self, x, label):
        #  time_major默认为True
        return tf.nn.ctc_loss(label, x, sequence_length=self.seq_len)

    def _build_network(self, input_shape, lr=1e-2, epoch=1e1, mode='train'):
        # 构建整个网络
        # 定义placeholder
        with tf.name_scope(name='ph'):
            self.X = tf.placeholder(tf.float32, shape=input_shape, name='x')
            self.y = tf.sparse_placeholder(tf.int32, name='y')  # 不指定shape时，可以feed任意shape
            self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

            self.keep_prob = tf.placeholder(tf.float32, name='kp')

        # 定义CNN kernels
        with tf.name_scope(name='cnn_kernels'):
            self.w00 = self._init_variable([3, 3, 1, 128], name='conv_w00')
            self.w01 = self._init_variable([3, 3, 128, 128], name='conv_w01')

            self.w10 = self._init_variable([3, 3, 128, 256], name='conv_w10')
            self.w11 = self._init_variable([3, 3, 256, 256], name='conv_w11')

            self.w20 = self._init_variable([3, 3, 256, 512], name='conv_w20')
            self.w21 = self._init_variable([3, 3, 512, 512], name='conv_w21')

        with tf.name_scope('architecture'):
            self.head = self.image2head(self.X)
            self.output = self.head2tail(self.head)  # self.output == self.tail

    def architecture(self, input_shape, lr=1e-2, epoch=1e1, mode='train'):
        # 构建图
        with tf.Graph().as_default():
            self.FLAGS = tf.flags.FLAGS

            config = tf.ConfigProto(
                allow_soft_placement=self.FLAGS.allow_soft_placement,  # 设置让程序自动选择设备运行
                log_device_placement=self.FLAGS.log_device_placement)

            self.sess = tf.Session(config=config)

            with self.sess.as_default():
                self._build_network(input_shape, lr=lr, epoch=epoch, mode=mode)

                # 计算误差
                with tf.name_scope('loss'):
                    self.loss = self.ctc_loss(self.output, self.y)

                # 使用编辑距离计算准确率
                with tf.name_scope('accuracy'):
                    #  time_major默认为True
                    self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.output,
                                                                                self.seq_len,
                                                                                merge_repeated=False)

                    first_dim = 0
                    second_dim = 0
                    concat_indices = None
                    concat_values = None
                    for i in range(len(self.decoded)):
                        decoded = self.decoded[i]
                        first_dim += decoded.shape[0]
                        if i == 0:
                            concat_indices = decoded.indices
                            concat_values = decoded.values
                        else:
                            concat_indices = tf.concat([concat_indices, decoded.indices], axis=0)
                            concat_values = tf.concat([concat_values, decoded.values], axis=0)

                        first_dim = self.FLAGS.batch_size
                        second_dim = tf.reduce_max(self.seq_len)

                        concat_values = tf.cast(concat_values, tf.int32)

                    print('Informations  \noutput:{}\ndecoded:{},\nseq_len:{},'
                          '\nconcat_indices:{},\nconcat_values:{}'
                          .format(self.output, self.decoded, self.seq_len, concat_indices, concat_values))
                    print('first dimension:{},\nsecond dimension:{}'.format(first_dim, second_dim))
                    decoded_tensor = tf.SparseTensor(indices=concat_indices, values=concat_values,
                                                     dense_shape=[first_dim, second_dim])

                    edit_distance = tf.edit_distance(decoded_tensor, self.y, name='edit_distance')
                    self.acc = tf.subtract(tf.constant(1, dtype=tf.float32), tf.reduce_mean(edit_distance), name='subtract')
                    self.acc_op = tf.identity(self.acc)

                self.learning_rate = 1e-3

                self.global_step = tf.Variable(0, name='global_step', trainable=True)

                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

                # 开始记录信息
                self.summary()

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
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def _accuracy(self):
        test_inputs, test_targets, test_seq_len = get_next_batch(self.FLAGS.batch_size)
        test_feed = {self.X: test_inputs,
                     self.y: test_targets,
                     self.seq_len: test_seq_len,
                     self.keep_prob: self.FLAGS.dropout_keep_prob}
        dd, log_probs, accuracy = self.sess.run([self.decoded[0], self.log_prob, self.acc_op],
                                                feed_dict=test_feed)
        self.calc_accuracy(dd, test_targets)

    def summary(self):
        # Summary
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, '../../', 'run_temp', timestamp))
        print('Writing to {}\n'.format(out_dir))

        tf.print('output result:', self.output)
        # Summary for loss and accuracy
        loss_summary = tf.summary.scalar('loss_summ', tf.reduce_max(self.loss, name='reduce_max_loss'))
        acc_summary = tf.summary.scalar('accuracy_summ', self.acc_op)

        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Train summaries
        # self.train_summary_op = tf.contrib.deprecated.merge_summary([loss_summary, acc_summary])
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        # self.train_summary_writer = tf.contrib.summary.SummaryWriter(train_summary_dir, self.sess.graph_def)
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

        # Dev summaries
        # self.dev_summary_op = tf.contrib.deprecated.merge_summary([loss_summary, acc_summary])
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # self.dev_summary_writer = tf.contrib.summary.SummaryWriter(dev_summary_dir, self.sess.graph_def)
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

    def checkpoint(self, out_dir):
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver = tf.train.Saver(tf.all_variables())
        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.global_step)
        print("Saved model checkpoint to {}\n".format(path))

    def train_step(self, inputs, sparse_targets, seq_len):
        feed_dict = {
            self.X: inputs,
            self.y: sparse_targets,
            self.keep_prob: self.FLAGS.dropout_keep_prob,
            self.seq_len: seq_len
        }

        _, step, summaries, loss, accuracy = self.sess.run(
            [self.train_op, self.global_step,
             self.train_summary_op, self.loss, self.acc_op
             ],
            feed_dict=feed_dict
        )

        time_str = datetime.datetime.now().isoformat()
        print("{}:step{},loss:{},acc:{},decoded:{}".format(time_str, step, loss, accuracy, self.decoded[0]))

        # self.train_summary_writer.add_summary(summaries, step)
        if step % self.FLAGS.batch_size == 0:
            print('epoch:{}'.format(step // self.FLAGS.batch_size))

    def dev_step(self, inputs, sparse_targets, seq_len):
        feed_dict = {
            self.X: inputs,
            self.y: sparse_targets,
            self.keep_prob: 1.,
            self.seq_len: seq_len
        }

        _, step, summaries, loss, accuracy = self.sess.run(
            [self.train_op, self.global_step,
             self.dev_summary_op, self.loss, self.acc_op
             ],
            feed_dict=feed_dict
        )

    def run(self):
        for epoch in range(100):
            for step in range(128):
                inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
                print('sequence length', seq_len)
                self.train_step(inputs, sparse_targets, seq_len)
                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % self.FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
                    self.dev_step(inputs, sparse_targets, seq_len)
                    print('Evaluation Done\n')
                    self._accuracy()


if __name__ == '__main__':
    tf.app.flags.DEFINE_boolean("allow_soft_placement",
                                True, "Allow device soft device placement")
    tf.app.flags.DEFINE_boolean("log_device_placement",
                                False, "Log placement of ops on devices")
    tf.app.flags.DEFINE_integer("batch_size",
                                64, "Batch Size (default: 64)")
    tf.app.flags.DEFINE_float("dropout_keep_prob",
                              0.85, "Dropout keep probability (default: 0.5)")
    tf.app.flags.DEFINE_integer("evaluate_every",
                                10, "Evaluate model on dev set after this many steps (default: 100)")
    tf.app.flags.DEFINE_integer('rnn_units',
                                128, "Rnn Units")
    crnn = CRNN(11)
    crnn.architecture(input_shape=[None, 400, 80, 1])
    print('Build model done!')
    crnn.run()

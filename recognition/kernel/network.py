# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
import datetime
import time
from recognition.kernel.data_provider import *

DIGITS = '0123456789'
# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 280, 28, 8, len(characters) + 1


class CRNN():
  """

  """

  def __init__(self, num_class):
    self.num_class = num_class
    self.interval_loss = 100
    self.FLAGS = tf.flags.FLAGS
    self.batch_size = self.FLAGS.batch_size

  def _init_variable(self, shape, name=None):
    """
    Create and initialize tensorflow variable.
    Args:
        shape: A 1-D integer Tensor or Python array. The shape of the initiated variable.
        name: A string. The name in tensorflow graph of the initiated variable.
    Return: A variable.
    """
    return tf.Variable(tf.truncated_normal(shape, mean=0., stddev=1.), name=name)

  def image2head(self, x):
    for i in range(3):
      # print('Defind layer use kernel {} with name {}'.format('self.w%s0' % i, 'cnn0%s' % i))
      x = tf.nn.conv2d(x, eval('self.w%s0' % i), [1, 1, 1, 1],
                       padding='SAME', name='cnn0%s' % i)
      x = tf.layers.batch_normalization(x, name='bn0%s' % i)
      tf.nn.relu(x)

      x = tf.nn.conv2d(x, eval('self.w%s1' % i), [1, 1, 1, 1],
                       padding='SAME', name='cnn1%s' % i)
      x = tf.layers.batch_normalization(x, name='bn1%s' % i)
      tf.nn.relu(x)

      print('Use the kernel:', eval('self.w%s0' % i), eval('self.w%s1' % i))

      x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='cnn2%s' % i)
    x = tf.nn.dropout(x, keep_prob=self.keep_prob, name='dropout')
    return x

  def head2tail(self, x):
    self.rnn_units = self.FLAGS.rnn_units
    shape = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, shape[1], shape[2] * shape[3]])
    self.w0 = self._init_variable(shape=[self.batch_size, shape[2] * shape[3], self.rnn_units], name='w0')
    self.b0 = self._init_variable(shape=[self.rnn_units, ], name='b0')
    x = tf.nn.xw_plus_b(x, self.w0, self.b0, name='dense0')
    # x = slim.fully_connected(x, self.rnn_units,
    #                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    x = tf.layers.batch_normalization(x, name='bn2')
    x = tf.nn.relu(x)

    # 构建双向叠加RNN
    # initial_state_fw = cell.zero_state(shape[0], dtype=tf.float32)
    # initial_state_bw = back_cell.zero_state(shape[0], dtype=tf.float32)

    cell = rnn.GRUCell(self.rnn_units, name='frnn', reuse=tf.AUTO_REUSE,
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       activation=tf.nn.tanh)
    back_cell = rnn.GRUCell(self.rnn_units, name='brnn', reuse=tf.AUTO_REUSE,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            activation=tf.nn.tanh)

    x, _ = tf.nn.bidirectional_dynamic_rnn(cell, back_cell, x, self.seq_len,
                                           dtype=tf.float32, time_major=False)

    x = tf.add(x[0], x[1], name='add')

    cell = rnn.GRUCell(self.rnn_units, name='frnn1', reuse=tf.AUTO_REUSE,
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       activation=tf.nn.tanh)
    back_cell = rnn.GRUCell(self.rnn_units, name='brnn1', reuse=tf.AUTO_REUSE,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            activation=tf.nn.tanh)

    # 构建双向拼接RNN
    x, _ = tf.nn.bidirectional_dynamic_rnn(cell, back_cell, x, self.seq_len,
                                           # initial_state_fw,
                                           # initial_state_bw,
                                           dtype=tf.float32, time_major=False)

    x = tf.concat([x[0], x[1]], axis=-1, name='concat')

    # # 构建多层RNN
    # fw_cell_list = [rnn.GRUCell(nh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    #                 for nh in [self.rnn_units] * 8]
    # # Backward direction cells
    # # bw_cell_list = [rnn.GRUCell(nh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # #                 for nh in [self.rnn_units] * 2]
    # # x, _, _ = rnn.stack_bidirectional_dynamic_rnn(
    # #   fw_cell_list, bw_cell_list, x, sequence_length=self.seq_len, dtype=tf.float32)
    #
    # cell = rnn.MultiRNNCell(fw_cell_list)
    # x, _ = tf.nn.dynamic_rnn(cell, x, self.seq_len, dtype=tf.float32)

    self.w1 = self._init_variable(shape=[self.batch_size, self.rnn_units * 2, self.num_class], name='w1')
    self.b1 = self._init_variable(shape=[self.num_class, ], name='b1')
    x = tf.nn.xw_plus_b(x, self.w1, self.b1, name='dense1')
    x = tf.nn.softmax(x, name='bottom_softmax')

    # x = slim.fully_connected(x, self.num_class, activation_fn=tf.nn.softmax)

    # time major 模式需要的input shape:(max_time x batch_size x num_classes)
    x = tf.transpose(x, (1, 0, 2))

    return x

  def _change_size(self, input_shape, channel=None, auto_change_channel=True):
    """
    变换特征图尺寸，CNN中使用
    Args:
      input_shape:
      channel:
      auto_change_channel:
    :return: 改变后的尺寸
    """
    input_shape[2] = input_shape[3]
    if auto_change_channel:
      input_shape[3] = input_shape[3] * 2
    if channel:
      input_shape[3] = channel
    return input_shape

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

      kernel_shape = [3, 3, 1, 32]
      self.w00 = self._init_variable(kernel_shape, name='conv_w00')
      for i in range(3):
        for j in range(2):
          if i == 0 and j == 0:
            continue
          auto_change_channel = True if j < 1 else False
          kernel_shape = self._change_size(kernel_shape, auto_change_channel=auto_change_channel)
          self.__dict__['w%s%s' % (i, j)] = self._init_variable(kernel_shape, name='conv_w%s%s' % (i, j))

    with tf.name_scope('architecture'):
      self.head = self.image2head(self.X)
      # self.head = self.X
      self.output = self.head2tail(self.head)  # self.output == self.tail

  def architecture(self, input_shape, lr=1e-2, epoch=1e1, mode='train'):
    # 构建图
    # with tf.Graph().as_default():

    config = tf.ConfigProto(
      allow_soft_placement=self.FLAGS.allow_soft_placement,  # 设置让程序自动选择设备运行
      log_device_placement=self.FLAGS.log_device_placement)

    self.sess = tf.Session(config=config)

    with self.sess.as_default():
      self._build_network(input_shape, lr=lr, epoch=epoch, mode=mode)
      # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

      with tf.name_scope('accuracy'):
        # 计算误差
        #  time_major默认为True
        self.loss = tf.reduce_mean(
          tf.nn.ctc_loss(labels=self.y, inputs=self.output,
                         sequence_length=self.seq_len, preprocess_collapse_repeated=True))

        #  time_major默认为True
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.output, self.seq_len,
                                                                    merge_repeated=False, top_paths=1)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.output, self.seq_len)

        concat_indices = None
        concat_values = None
        accelerate_indices = 0

        for i in range(len(self.decoded)):
          decoded = self.decoded[i]
          if i == 0:
            concat_indices = decoded.indices
            concat_values = decoded.values
          else:
            decoded_indices = tf.concat([tf.expand_dims(decoded.indices[:, 0] + accelerate_indices, axis=-1),
                                         tf.expand_dims(decoded.indices[:, 1], axis=-1)], axis=1)
            concat_indices = tf.concat([concat_indices, decoded_indices], axis=0)
            concat_values = tf.concat([concat_values, decoded.values], axis=0)

          accelerate_indices += decoded.shape[0]

        first_dim = self.FLAGS.batch_size
        second_dim = tf.reduce_max(self.seq_len)

        concat_values = tf.cast(concat_values, tf.int32)

        print('Informations  \noutput:{}\ndecoded:{},\nseq_len:{},'
              '\nconcat_indices:{},\nconcat_values:{}'
              .format(self.output, self.decoded, self.seq_len, concat_indices, concat_values))
        print('first dimension:{},\nsecond dimension:{}'.format(first_dim, second_dim))

        decoded_tensor = tf.SparseTensor(indices=concat_indices, values=concat_values,
                                         dense_shape=[first_dim, second_dim])

        # 使用编辑距离计算准确率
        edit_distance = tf.edit_distance(decoded_tensor, self.y, name='edit_distance')
        self.acc_op = tf.subtract(tf.constant(1, dtype=tf.float32), tf.reduce_mean(edit_distance), name='subtract')

      self.global_step = tf.Variable(0, name='global_step', trainable=True)
      self.learning_rate = tf.train.exponential_decay(self.FLAGS.INITIAL_LEARNING_RATE,
                                                      self.global_step,
                                                      self.FLAGS.DECAY_STEPS,
                                                      self.FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                                      staircase=True)

      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
      self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, self.global_step)

      # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
      #   .minimize(self.loss, global_step=self.global_step)

      # 开始记录信息
      self.summary()

      self.sess.run(tf.global_variables_initializer())

  def calc_accuracy(self, decode_list, test_target):
    original_list = decode_sparse_tensor(test_target)
    detected_list = decode_sparse_tensor(decode_list)
    # detected_list = self.sess.run(tf.sparse_to_dense(decode_list))
    true_numer = 0

    if len(original_list) != len(detected_list):
      print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
            " test and detect length desn't match")
      return
    print("T/F: original(length) <-------> detected(length)")
    for idx, number in enumerate(original_list):
      detect_number = detected_list[idx]
      hit = (number == detect_number)
      print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
      if hit:
        true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

  def _accuracy(self, test_inputs, test_targets, test_seq_len):
    # test_inputs, test_targets, test_seq_len = get_next_batch(self.FLAGS.batch_size)
    # test_inputs, test_targets, test_seq_len = get_next_batch(1)
    print('test sequence length:', test_seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.output,
                                                      test_seq_len,
                                                      merge_repeated=False)

    # decoded, log_prob = tf.nn.ctc_greedy_decoder(self.output,
    #                                              test_seq_len)

    test_feed = {self.X: test_inputs,
                 self.y: test_targets,
                 self.seq_len: test_seq_len,
                 self.keep_prob: 1.}
    dd, log_probs = self.sess.run([decoded[0], log_prob],
                                  feed_dict=test_feed)
    self.calc_accuracy(dd, test_targets)

  def summary(self):
    # Summary
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, '../../', 'run_temp', timestamp))
    print('Writing to {}\n'.format(out_dir))

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
    self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

    # Dev summaries
    self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
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

    # time_str = datetime.datetime.now().isoformat()
    # print("{}:step{},loss:{},acc:{},decoded:{}".format(time_str, step, loss, accuracy, self.decoded[0]))
    if self.interval_loss > loss:
      self.interval_loss = loss
      self.sess.run(tf.assign(self.global_step, tf.add(self.global_step, 1)))
      print("step:{}\tloss:{}\tacc:{}".format(step, loss, accuracy))
    print("step:{}\tloss:{}\tacc:{}".format(step, loss, accuracy))

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
    inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
    for epoch in range(128):
      inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
      for step in range(32):
        inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
        # plot(inputs[0], decode_sparse_tensor(sparse_targets)[0])
        # print('sequence length', seq_len)
        self.train_step(inputs, sparse_targets, seq_len)
        current_step = tf.train.global_step(self.sess, self.global_step)
        if current_step % self.FLAGS.evaluate_every == 0:
          print("\nAfter epoch %s Evaluation:" % epoch)
          # inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
          self.dev_step(inputs, sparse_targets, seq_len)
          print('Evaluation Done\n')
          self._accuracy(inputs, sparse_targets, seq_len)
      # print("\nAfter epoch %s Evaluation:" % epoch)
      # # inputs, sparse_targets, seq_len = get_next_batch(self.FLAGS.batch_size)
      # self.dev_step(inputs, sparse_targets, seq_len)
      # print('Evaluation Done\n')
      # self._accuracy()


if __name__ == '__main__':
  tf.app.flags.DEFINE_boolean("allow_soft_placement",
                              True, "Allow device soft device placement")
  tf.app.flags.DEFINE_boolean("log_device_placement",
                              False, "Log placement of ops on devices")
  tf.app.flags.DEFINE_integer("batch_size",
                              32, "Batch Size (default: 64)")
  tf.app.flags.DEFINE_float("dropout_keep_prob",
                            0.75, "Dropout keep probability (default: 0.5)")
  tf.app.flags.DEFINE_integer("evaluate_every",
                              10, "Evaluate model on dev set after this many steps (default: 100)")
  tf.app.flags.DEFINE_integer('rnn_units',
                              128, "Rnn Units")
  # 初始化学习速率
  tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 1e-4, 'Learning rate initial value')
  tf.app.flags.DEFINE_integer('DECAY_STEPS', 5000, 'DECAY_STEPS')
  tf.app.flags.DEFINE_integer('REPORT_STEPS', 100, 'REPORT_STEPS')
  tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.9, 'LEARNING_RATE_DECAY_FACTOR')

  crnn = CRNN(n_class)
  crnn.architecture(input_shape=[None, width, height, 1])
  print('Build model done!')
  crnn.run()
  print('Training done!')

# coding:utf8
import tensorflow as tf
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *

import sys

from recognition.kernel.data_helper import GenSequenceChars
# from data_helper import GenSequenceChars
from tensorflow.python import debug as tf_debug


class SequenceCharsModel(object):
  def __index__(self):
    """
    Initialize the recognition model.
    Args:
    Returns:

    """

  def ctc_lambda_func(self, args):
    """
    CTC　loss function.
    Args:
      args: A list contains y_pred, labels, input_length, label_length, all of them are ndarray.

    Returns:
      A Tensor of ctc_cost.
    """
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]  # [batch, rnn_step(height), num_class]
    # input_length means sequence length of inputs, each of the input data may has different length
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

  def _build_network(self, tfrecord=None):
    """
    Build main network structure
    Args:
      tfrecord: A list contains input_batch, label_batch, all of them are tfrecord train batch data.

    Returns:

    """
    if not tfrecord:
      input_tensor = Input(name='input_tensor', shape=self.input_size)
      labels = Input(name='the_labels', shape=[self.n_len], dtype='float32')

      input_length = Input(name='input_length', shape=(1,), dtype='int64')  # length of input sequence tensor
      label_length = Input(name='label_length', shape=(1,), dtype='int64')  # length of output sequence tensor
    else:
      print('tfrecord shape', tfrecord[0].shape, tfrecord[1].shape, tfrecord[2].shape, tfrecord[3].shape)

      # 使用tfrecord接入Input
      input_tensor = Input(tensor=tfrecord[0], batch_shape=tfrecord[0].get_shape().as_list())

      labels = Input(name='the_labels', tensor=tfrecord[1],
                     batch_shape=tfrecord[1].get_shape().as_list(), dtype='int32')

      input_length = Input(name='input_length',
                           tensor=tf.reshape(tfrecord[2], [-1, 1]),
                           batch_shape=tfrecord[2].get_shape().as_list() + [1],  # batch_shape参数会reshape 传入的tensor
                           dtype='int64')  # 输入序列的Tensor长度

      label_length = Input(name='label_length',
                           tensor=tf.reshape(tfrecord[3], [-1, 1]),
                           batch_shape=tfrecord[3].get_shape().as_list() + [1],
                           dtype='int64')  # 输出序列的Tensor长度

    x = input_tensor

    # 6层CNN
    for i in range(3):
      x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal', padding='SAME')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal', padding='SAME')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape().as_list()
    self.rnn_step = conv_shape[1]
    self.rnn_dim = conv_shape[2] * conv_shape[3]

    # Convert tensor shape to [batch, rnn_step, any]
    x = Reshape(target_shape=(self.rnn_step, self.rnn_dim))(x)  # [batch, height, width * units]

    # Flatten tensor
    x = Dense(self.rnn_units, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bi-RNN and add
    gru_1 = GRU(self.rnn_units, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(self.rnn_units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])  # [batch, height, units]

    # Bi-RNN and concatenate
    gru_2 = GRU(self.rnn_units, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(self.rnn_units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 name='gru2_b')(gru1_merged)

    x = concatenate([gru_2, gru_2b])  # [batch, height, units * 2]
    x = Dropout(self.drop_prob)(x)
    # Full connection convert tensor shape to [batch, height, n_class]
    x = Dense(self.num_class, kernel_initializer='he_normal', activation='softmax')(x)
    # self.base_model = Model(input=input_tensor, output=x)  # [batch, height, units * 2]
    self.base_model = Model(inputs=input_tensor, outputs=x)  # [batch, height, units * 2]
    # self.base_model = make_parallel(base_model, 4)

    print('base output', self.base_model.output.shape)
    print('labels', labels.shape)

    # 把CTC_LOSS当成layer而非loss, 则在CTC下一层做一个跟标签匹配的output
    loss_out = Lambda(self.ctc_lambda_func, name='ctc')([self.base_model.output, labels, input_length, label_length])
    print('loss out', loss_out)
    self.model = Model(inputs=(input_tensor, labels, input_length, label_length), outputs=loss_out)

  def _architecture(self, num_class, drop_prob, n_len, input_size, opti='adam', rnn_units=128):
    """
    Construct and compile model.
    Args:
      num_class: A int value, number of categories.
      drop_prob: A float value, rate of you wanna throw.
      n_len: A int value, sequence length.
      input_size: A tuple contains height, width and dimension.
      opti: A string, optimization method.
      rnn_units: A int, number of rnn output units.

    Returns:

    """
    self.drop_prob = drop_prob
    self.num_class = num_class
    self.n_len = n_len
    self.rnn_units = rnn_units
    self.input_size = input_size

    # Build network
    self._build_network()

    # 走个形势
    self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opti)

  def architecture_tfrecord(self, num_class, drop_prob, n_len,
                            input_size, batch_size, tfrecord_path, opti='adam', rnn_units=128):
    """

    Args:
      num_class: A int value, number of categories.
      drop_prob: A float value, rate of you wanna throw.
      n_len: A int value, sequence length.
      input_size: A tuple contains height, width and dimension.
      batch_size:
      tfrecord_path:
      opti: A string, optimization method.
      rnn_units: A int, number of rnn output units.

    Returns:

    """
    self.drop_prob = drop_prob
    self.num_class = num_class
    self.n_len = n_len
    self.rnn_units = rnn_units
    self.input_size = input_size
    self.batch_size = batch_size

    # 获取Session
    # self.sess = K.get_session()
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                 log_device_placement=True))
    # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    # 读取用于训练的TFRecord数据集
    self.data = tf.data.TFRecordDataset(tfrecord_path)
    self.train_batch = self.data.shuffle(1048576).map(self._read_features)
    # self.label_batch = self.data.map(self._read_labels)

    self.dataset = self.train_batch.batch(self.batch_size).repeat(1024)

    self.iterator = self.data.make_one_shot_iterator()
    self.next_element = self.iterator.get_next()

    # filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=False)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    #
    # image, label, input_length, label_length, __ = self._read_features(serialized_example)
    #
    # self.image_batch, self.label_batch, self.input_length_batch, self.label_length_batch, ___batch = \
    #   tf.train.batch([image, label, input_length, label_length, __], batch_size=batch_size)
    #
    # self.label_batch = tf.string_split(self.label_batch, delimiter=',')
    # self.label_batch = tf.sparse_tensor_to_dense(self.label_batch, default_value='')

    # Build network
    # self._build_network(tfrecord=[image_batch, label_batch, input_length_batch, label_length_batch, ___batch])

    self._build_network()

    # 走个形势
    self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opti)

  def run(self, train_path, val_path, batch_size, model_path=None,
          step_per_epoch=100, epochs=20, gene=1, save_when_percent=80):
    """

    Args:
      train_path:
      val_path:
      batch_size:
      model_path:
      step_per_epoch:
      epochs:
      gene:
      save_when_percent:

    Returns:

    """
    # base_path, batch_size, input_size, n_len, rnn_step, gene
    self.train_gen = GenSequenceChars(train_path, batch_size, self.input_size,
                                      self.n_len, rnn_step=self.rnn_step, gene=gene)

    self.val_gen = GenSequenceChars(val_path, batch_size, self.input_size,
                                    self.n_len, rnn_step=self.rnn_step, gene=gene)

    # model, n_len, model_path, gen, batch_size=128, steps=20, save_when_percent=80
    self.evaluator = Evaluator(self.base_model, self.model, self.n_len, model_path, self.val_gen, batch_size=batch_size,
                               save_when_percent=save_when_percent)

    self.history = self.model.fit_generator(self.train_gen.gen(), steps_per_epoch=step_per_epoch, epochs=epochs,
                                            callbacks=[self.evaluator],
                                            validation_data=self.val_gen.gen(), validation_steps=20)
    if model_path:
      self.save(os.path.join(model_path, '%s%s' % ('finally', '.h5')))

  def run_multi_thread(self, batch_size, val_path, model_path=None,
                       steps_per_epoch=100, epochs=20, save_when_percent=80):
    """

    Args:
      batch_size:
      val_path:
      model_path:
      steps_per_epoch:
      epochs:
      save_when_percent:

    Returns:

    """
    self.batch_size = batch_size

    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(self.sess, coord)

    try:
      while not coord.should_stop():
        self.val_gen = GenSequenceChars(val_path, batch_size, self.input_size,
                                        self.n_len, rnn_step=self.rnn_step, gene=1)

        self.evaluator = Evaluator(self.base_model, self.model, self.n_len, model_path, self.val_gen,
                                   batch_size=batch_size,
                                   save_when_percent=save_when_percent)

        print('开始训练')
        # self.history = self.model.fit(epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[self.evaluator])
        print('train_batch', self.train_batch)
        self.history = self.model.fit(self.dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                      callbacks=[self.evaluator])
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()
    coord.join(threads)

    if model_path:
      self.save(os.path.join(model_path, '%s_%s' % ('finally', '.h5')))

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()

  def _read_features(self, example_proto):
    """

    :param example_proto:
    :return:
    """
    dic = dict()
    dic['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
    dic['label'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
    dic['input_length'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    dic['label_length'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    dic['__'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)

    parse_example = tf.parse_single_example(
      serialized=example_proto, features=dic)

    # img = parse_example['image']
    y = parse_example['label']

    # 船新添加
    y = tf.string_split(tf.expand_dims(y, -1), delimiter=',')
    y = tf.sparse.to_dense(y, default_value='-1')
    # y = tf.sparse_tensor_to_dense(y, default_value='-1')
    y = tf.squeeze(tf.string_to_number(y), 0)

    input_length = parse_example['input_length']
    label_length = parse_example['label_length']
    __ = parse_example['__']
    # __ = tf.cast(tf.string_to_number(parse_example['__']), tf.int64)

    img = tf.decode_raw(parse_example['image'], out_type=tf.float64)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [self.input_size[0], self.input_size[1], 1])

    return (img, y, input_length, label_length), __
    # return img, y, input_length, label_length

  def save(self, model_path):
    """
    Save the trained model.
    Args:
      model_path:

    Returns:

    """
    self.base_model.save(model_path)


class Evaluator(Callback):
  """
  Evaluator for some model.
  """

  def __init__(self, base_model, model, n_len, model_path, gen, batch_size=128, steps=20, save_when_percent=80):
    super(Evaluator).__init__()
    self.accs = []
    self.base_model = base_model
    self.model = model
    self.n_len = n_len
    self.model_path = model_path
    self.gen = gen
    self.batch_size = batch_size
    self.steps = steps
    self.save_when_percent = save_when_percent

  def on_epoch_end(self, epoch, logs=None):
    """
    Callbatch when every epoch has been done.
    Args:
      epoch:
      logs:

    Returns:

    """
    acc = self.evaluate(steps=self.steps) * 100
    self.accs.append(acc)
    if acc >= self.save_when_percent:
      self.save()
      print('Temporary model saved!')
    print('acc: %f%%' % acc)

  def evaluate(self, steps=10):
    """
    Evaluate current model's accuracy.
    Args:
      steps:

    Returns:

    """
    batch_acc = 0
    generator = self.gen.gen()
    for i in range(steps):
      (X_test, y_test, _, _), _ = next(generator)
      y_pred = self.base_model.predict(X_test)
      shape = y_pred[:, :, :].shape
      ctc_decode = K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
      out = K.get_value(ctc_decode)[:, :self.n_len]
      print(y_test[0], out[0])
      if out.shape[1] == self.n_len:
        batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps

  def save(self, mode='epoch'):
    """
    Save the trained model.
    Args:
      mode:

    Returns:

    """
    if mode == 'epoch':
      base_model_path = os.path.join(self.model_path, '%s%s' % ('epoch', '.h5'))
      model_path = os.path.join(self.model_path, '%s%s' % ('epoch', '.h5'))
    else:
      base_model_path = os.path.join(self.model_path, '%s%s' % ('finally', '.h5'))
      model_path = os.path.join(self.model_path, '%s%s' % ('finally', '.h5'))
    self.base_model.save(base_model_path)
    self.model.save(model_path)


if __name__ == '__main__':
  scm = SequenceCharsModel()

  # 41配置
  # scm._architecture(num_class=27, drop_prob=0.2, n_len=10, input_size=[280, 48, 1], opti='adam', rnn_units=128)
  # scm.run(r'D:\data\cv\letters', r'D:\data\cv\letters', 128,
  #         model_path=r'D:\PycharmProjects\work\ocr\ocr\model_test',
  #         step_per_epoch=100, epochs=100, gene=1, save_when_percent=80)

  # scm.run('/home/lianxiaolei/sequence_recognition/letters', '/home/lianxiaolei/sequence_recognition/letters', 32,
  #         model_path='/home/lianxiaolei/sequence_recognition/model_test/',
  #         step_per_epoch=128, epochs=100, gene=4, save_when_percent=70)

  # 本机配置
  scm.architecture_tfrecord(num_class=27, drop_prob=0.2, n_len=10, input_size=[480, 48, 1],
                            batch_size=128, tfrecord_path=r'D:\data\cv\letters_480_48_1_10k.tfrecord',
                            opti='adam', rnn_units=128)
  scm.run_multi_thread(128, val_path=r'D:\data\cv\letters',
                       model_path=r'D:\PycharmProjects\work\ocr\ocr\model_test',
                       steps_per_epoch=128, epochs=20)

  print('All done.')

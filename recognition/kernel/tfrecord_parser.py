import tensorflow as tf
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
# from recognition.kernel.data_provider import *

# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 280, 28, 10, len(characters) + 1


def generate():
  ds = '0123456789'
  ts = ['{}{}{}{}{}', '({}{}{}){}{}', '{}{}({}{}{})']
  os = '+-*/'
  # os = ['+', '-', 'times', 'div']
  cs = [random.choice(ds) if x % 2 == 0 else random.choice(os) for x in range(5)]
  return random.choice(ts).format(*cs)


def get_img_by_char(char, base_path):
  """
  get a img by giving char
  :param char:
  :param base_path:
  :return:
  """
  opdict = {'+': 10, '-': 11, '*': 12, '/': 13, '=': 14, '(': 15, ')': 16}
  if char in opdict.keys():
    char = opdict[char]
  path = os.path.join(base_path, str(char))
  files = os.listdir(path)

  rdm = random.randint(0, len(files) - 1)

  if rdm >= len(files):
    print(path, len(files), rdm)

  file = files[rdm]
  path = os.path.join(path, file)
  return cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255


def get_sequence_img(chars, base_path):
  """

  :param chars:
  :return:
  """
  x = get_img_by_char(chars[0], base_path=base_path)
  for i in range(1, len(chars)):
    x = np.hstack([x, get_img_by_char(chars[i], base_path=base_path)])
  x = cv2.resize(x, (width, height))
  # x = skimage.util.random_noise(x, mode='gaussian', clip=True)
  return x


def gen(base_path):
  """

  :return:
  """
  random_str = ''.join([random.choice(characters) for j in range(n_len)])

  # img = np.array(get_sequence_img(random_str, base_path=base_path))
  # convert img to uint8
  # img = img.astype(np.uint8)
  # img = img.transpose(1, 0, 2)

  tmp = np.array(get_sequence_img(random_str, base_path=base_path))
  tmp = tmp.reshape([tmp.shape[0], tmp.shape[1], 1])
  tmp = tmp.transpose(1, 0, 2)
  img = tmp.astype(np.float64)

  return img, [characters.find(x) for x in random_str]


def _bytes_feature(value):
  """

  :param value:
  :return:
  """
  img_bytes = value.tostring()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))


def _float32_feature(value):
  """

  :param value:
  :return:
  """
  return tf.train.Feature(float32_list=tf.train.FloatList(value=[value]))


def _bytes_feature_with_list(value):
  """

  :param value:
  :return:
  """
  value = [str(v) for v in value]
  return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[bytes(''.join(value), encoding='utf8')]))


def _write_features(features, tf_writer):
  tf_features = tf.train.Features(feature=features)
  tf_example = tf.train.Example(features=tf_features)
  tf_serialized = tf_example.SerializeToString()

  tf_writer.write(tf_serialized)


def tfrecord(path, batch_size=128):
  """
  Deprecated
  :param batch_size:
  :return:
  """
  with tf.python_io.TFRecordWriter(path=path) as tf_writer:
    for i in range(batch_size):
      features = {}
      img, y = gen()

      features['image'] = _bytes_feature(img)
      features['label'] = _bytes_feature_with_list(y)

      _write_features(features, tf_writer)


def img2tfrecord(path, batch_size, data_path):
  """
  :param path
  :param batch_size:
  :return:
  """
  with tf.python_io.TFRecordWriter(path=path) as tf_writer:
    for i in range(batch_size):
      features = {}
      img, y = gen(data_path)

      features['image'] = _bytes_feature(img)
      features['label'] = _bytes_feature_with_list(y)

      _write_features(features, tf_writer)

      if i % 1000 == 0:
        print('Now process %s items.' % i)


def _read_features(example_proto):
  """

  :param example_proto:
  :return:
  """
  dic = dict()
  dic['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
  dic['label'] = tf.FixedLenFeature(shape=[], dtype=tf.string)

  parse_example = tf.parse_single_example(
    serialized=example_proto, features=dic)

  # img = parse_example['image']
  y = parse_example['label']

  img = tf.decode_raw(parse_example['image'], out_type=tf.float64)
  img = tf.cast(img, tf.float32)
  img = tf.reshape(img, [width, height, 1])

  return img, y


def sparse_tuple_from(sequences, dtype=np.int32):
  """
  Create a sparse representention of x.
  Args:
      sequences: A list of lists of type dtype where each element is a sequence
      dtype: Type of SparseTensor
  Returns:
      A tuple with (indices, values, shape)
  """
  indices = []
  values = []

  for n, seq in enumerate(sequences):
    # print(n, len(seq))
    indices.extend(zip([n] * len(seq), range(len(seq))))
    values.extend(seq)

  indices = np.asarray(indices, dtype=dtype)
  values = np.asarray(values, dtype=dtype)
  shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

  return indices, values, shape


def tfrecord2img(path, epoch_batch_size=1):
  """

  :param epoch_batch_size:
  :return:
  """
  data = tf.data.TFRecordDataset(path)
  # print('Type of data.map', type(data.map(_read_features)))
  # print('Type of data.map.batch', type(data.map(_read_features).batch(epoch_batch_size)))
  # 调用传入的函数一条一条的解出数据最后组成batch
  data = data.map(_read_features).batch(epoch_batch_size)
  # data = data.map(_read_features).shuffle(epoch_batch_size)

  iterator = data.make_one_shot_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      while not coord.should_stop():
        img_tensor, l_tensor = next_element
        img, l = sess.run([img_tensor, l_tensor])
        print(img.shape)
        print(type(l[1]), l)
        print('-' * 80)
        y = list(map(lambda seq: seq.decode('utf8'), l))
        # y = [[int(num) for num in item] for item in y]
        # plt.imshow(img[1, :, :, 0])
        # plt.title(' '.join(y[1]))
        # plt.show()
        y = sparse_tuple_from(y, dtype=np.int32)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()
    coord.join(threads)

  return img, y


def to_img_test(path, epoch_batch_size=1):
  data = tf.data.TFRecordDataset(path)
  # print('Type of data.map', type(data.map(_read_features)))
  # print('Type of data.map.batch', type(data.map(_read_features).batch(epoch_batch_size)))
  # 调用传入的函数一条一条的解出数据最后组成batch
  data = data.map(_read_features).batch(epoch_batch_size)

  iterator = data.make_one_shot_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    img, y = sess.run(next_element)
    y = list(map(lambda seq: seq.decode('utf8'), y))
    y = [[int(num) for num in item] for item in y]
    y = sparse_tuple_from(y, dtype=np.int32)
    while True:
      try:
        img, y = sess.run(next_element)
        y = list(map(lambda seq: seq.decode('utf8'), y))
        y = [[int(num) for num in item] for item in y]
        print('Decoded y:', y)

      except Exception as e:
        break

  return img, y


if __name__ == '__main__':
  img2tfrecord(r'D:\data\cv\numbers_280_48_1_20k.tfrecord',
               batch_size=1024 * 20, data_path=r'D:\data\cv\nums')

  # import time
  # tmp_time = time.time()
  # for i in range(4):
  #   imgs, labels = tfrecord2img('/Users/imperatore/data/numbers_1.tfrecord', epoch_batch_size=128)
  #   print('Decode tfrecord to normal data with data shape:%s and label length:%s.' % (imgs.shape, labels))
  # print('time cost:', time.time() - tmp_time)

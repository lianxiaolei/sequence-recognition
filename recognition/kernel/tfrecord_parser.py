import tensorflow as tf
import os
import cv2
import numpy as np
import skimage
import random


# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 400, 80, 10, len(characters) + 1


def generate():
    ds = '0123456789'
    ts = ['{}{}{}{}{}', '({}{}{}){}{}', '{}{}({}{}{})']
    os = '+-*/'
    # os = ['+', '-', 'times', 'div']
    cs = [random.choice(ds) if x % 2 == 0 else random.choice(os) for x in range(5)]
    return random.choice(ts).format(*cs)


def get_img_by_char(char, base_path=r'F:\number_ok1'):
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
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_sequence_img(chars):
    """

    :param chars:
    :return:
    """
    x = get_img_by_char(chars[0])
    for i in range(1, len(chars)):
        x = np.hstack([x, get_img_by_char(chars[i])])
    x = cv2.resize(x, (400, 80))
    x = skimage.util.random_noise(x, mode='gaussian', clip=True)
    return x


def gen():
    """

    :return:
    """
    random_str = ''.join([random.choice(characters) for j in range(n_len)])

    img = np.array(get_sequence_img(random_str))
    # convert img to uint8
    img = img.astype(np.uint8)

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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(' '.join(value), encoding='utf8')]))


def _write_features(features, tf_writer):
    tf_features = tf.train.Features(feature=features)
    tf_example = tf.train.Example(features=tf_features)
    tf_serialized = tf_example.SerializeToString()

    tf_writer.write(tf_serialized)


def tfrecord(path, batch_size=128):
    """

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


def img2tfrecord(path, batch_size=128):
    """
    :param path
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

    img = tf.decode_raw(parse_example['image'], out_type=tf.uint8)

    return img, y


def tfrecord2img(path, epoch_batch_size=1):
    """

    :param epoch_batch_size:
    :return:
    """
    data = tf.data.TFRecordDataset(path)
    data = data.map(_read_features).batch(epoch_batch_size)

    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        img, y = sess.run(fetches=next_element)

        img = img.reshape(height, width)

        y = list(map(lambda x: int(x), y[0].decode('utf8').split()))

    return img, y


if __name__ == '__main__':
    img2tfrecord('./tfrc.tfrecord', batch_size=1)
    tf.summary.merge_all

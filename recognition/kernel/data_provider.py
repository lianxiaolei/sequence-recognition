# coding:utf8

import random
import os
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from recognition.kernel.sparse_parser import *
import matplotlib.pyplot as plt

# characters = '0123456789+-*/=()'
characters = '0_1_2_3_4_5_6_7_8_9_)_(_times_-_[_div_+_]_=_{_}'
inv_extend_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: ')', 11: '(',
                   12: 'times', 13: '-', 14: '[', 15: 'div', 16: '+', 17: ']', 18: '=', 19: '{', 20: '}'}
char_list = characters.split('_')
width, height, n_len = 256, 64, 6


# width, height, n_len = 256, 32, 6

# datagen = ImageDataGenerator(
#   rotation_range=0.4,
#   width_shift_range=0.04,
#   height_shift_range=0.04,
#   shear_range=0.2,
#   zoom_range=0.0,
#   fill_mode='nearest')


def plot(img, title):
  plt.imshow(img[:, :, 0])
  plt.title(title)
  plt.show()


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
  # opdict = {')': 11, '(': 12, 'times': 13, '-': 14, '[': 15, 'div': 16, '+': 17, ']': 18, '=': 19, '{': 20, '}': 21}

  # if char in opdict.keys():
  #   char = opdict[char]
  path = os.path.join(base_path, str(char))
  files = os.listdir(path)

  rdm = random.randint(0, len(files) - 1)

  if rdm >= len(files):
    print(path, len(files), rdm)

  file = files[rdm]
  path = os.path.join(path, file)
  return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_sequence_img(chars, base_path):
  x = get_img_by_char(chars[0], base_path=base_path)
  for i in range(1, len(chars)):
    x = np.hstack([x, get_img_by_char(chars[i], base_path=base_path)])
  x = cv2.resize(x, (width, height))
  # x = skimage.util.random_noise(x, mode='gaussian', clip=True)
  return x


def get_next_batch(batch_size=128, base_path=None, gene=1):
  X = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
  # X = np.zeros((batch_size, width, height), dtype=np.uint8)
  y = np.zeros((batch_size, n_len), dtype=np.uint8)
  for i in range(batch_size):
    random_str = [random.choice(char_list) for j in range(n_len)]
    # random_str = ''.join([random.choice(char_list) for j in range(n_len)])
    #             random_str = '60/3=20'
    # print('random string', random_str)
    tmp = np.array(get_sequence_img(random_str, base_path=base_path))
    tmp = np.reshape(tmp, newshape=(tmp.shape[0], tmp.shape[1], 1))
    tmp = np.transpose(tmp, (1, 0, 2))

    X[i] = tmp
    y[i] = [char_list.index(x) for x in random_str]

    # plot(X[i], '{}'.format([inv_extend_dict[x] for x in y[i]]))

  # i = 0
  # XX = None
  # yy = None
  # for batch in datagen.flow(X, y, batch_size=batch_size):
  #   #             print(batch[0].shape, batch[1].shape)
  #
  #   if not type(XX) == np.ndarray:
  #     XX = batch[0]
  #     yy = batch[1]
  #   else:
  #     XX = np.concatenate([XX, batch[0]], axis=0)
  #     yy = np.concatenate([yy, batch[1]], axis=0)
  #
  #   i += 1
  #   if i >= gene:
  #     break

  sparse_target = sparse_tuple_from(y)
  # seq_len = np.ones(batch_size) * (n_len * 2 + 1)
  seq_len = np.ones(batch_size) * int(width / 8)

  return X / 255., sparse_target, seq_len


def extend_dict(base_path):
  extend_dict = {}
  inv_extend_dict = {}
  num_list = [i for i in '0123456789']
  signals = os.listdir(base_path)
  signals.remove('hidden')
  for num in num_list:
    signals.remove(num)
  sorted(signals)
  string = ''
  for i, s in enumerate(signals, 11):
    print(i, s)
    extend_dict[s] = i
    inv_extend_dict[i] = s
    string += '_%s' % s
  print(extend_dict)
  print(inv_extend_dict)
  print(string)


if __name__ == '__main__':
  extend_dict('/home/lian/data/cv/hand-written-math-symbol/images_rb')
  # res = get_next_batch(1, base_path='/home/lian/data/cv/hand-written-math-symbol/images_rb')

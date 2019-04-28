# coding:utf8

import random
import os
import cv2
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
from recognition.kernel.sparse_parser import *
import matplotlib.pyplot as plt

# characters = '0123456789+-*/=()'
characters = '0123456789'
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


def get_img_by_char(char, base_path=r'D:\data\cv\nums'):
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
  x = get_img_by_char(chars[0])
  for i in range(1, len(chars)):
    x = np.hstack([x, get_img_by_char(chars[i])])
  x = cv2.resize(x, (width, height))
  # x = skimage.util.random_noise(x, mode='gaussian', clip=True)
  return x


def get_next_batch(batch_size=128, gene=1):
  X = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
  # X = np.zeros((batch_size, width, height), dtype=np.uint8)
  y = np.zeros((batch_size, n_len), dtype=np.uint8)
  for i in range(batch_size):
    random_str = ''.join([random.choice(characters) for j in range(n_len)])
    #             random_str = '60/3=20'
    tmp = np.array(get_sequence_img(random_str))
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)
    tmp = tmp.transpose(1, 0, 2)

    # tmp = tmp.reshape(tmp.shape[0], tmp.shape[1])
    # tmp = tmp.transpose(1, 0)

    X[i] = tmp
    y[i] = [characters.find(x) for x in random_str]

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


if __name__ == '__main__':
  res = get_next_batch(1, 1)
  print(res)

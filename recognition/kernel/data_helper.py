# coding:utf8

import random
import cv2
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# characters = '0123456789'
characters = 'abcdefghigklmnopqrstuvwxyz'

# extradict = {'+': 10, '-': 11, '*': 12, '/': 13, '=': 14, '(': 15, ')': 16}
extradict = {'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22,
             'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34,
             'y': 35, 'z': 36}


class GenSequenceChars(object):
  """
  Data generator.
  """

  def __init__(self, base_path, batch_size, input_size, n_len, rnn_step, gene=1):
    """

    Args:
      characters:
      base_path:
      batch_size:
      input_size:
      n_len:
      gene:
    """
    self.characters = characters
    self.base_path = base_path
    self.batch_size = batch_size
    self.input_size = input_size
    self.n_len = n_len
    self.gene = gene
    self.rnn_step = rnn_step

    self.datagen = image.ImageDataGenerator(
      rotation_range=0.4,
      width_shift_range=0.04,
      height_shift_range=0.04,
      shear_range=0.2,
      zoom_range=0.0,
      fill_mode='nearest')

  def get_img_by_char(self, char):
    """
    Get a img by giving char.
    :param char:
    :param base_path:
    :return:
    """
    # todo 扩展0～9之外的标签
    if char in extradict.keys():
      char = extradict[char]
    path = os.path.join(self.base_path, str(char))
    files = os.listdir(path)

    rdm = random.randint(0, len(files) - 1)
    file = files[rdm]
    path = os.path.join(path, file)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

  def get_sequence_img(self, chars):
    """
    Generate char sequence on a image.
    Args:
      chars:

    Returns:

    """
    x = self.get_img_by_char(chars[0])
    for i in range(1, len(chars)):
      x = np.hstack([x, self.get_img_by_char(chars[i])])
    x = cv2.resize(x, tuple(self.input_size[:2]))
    return x

  def gen(self):
    """
    Main entrance of generator.
    Args:
      rnn_step:

    Returns:

    """
    X = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.uint8)
    y = np.zeros((self.batch_size, self.n_len), dtype=np.uint8)
    while True:
      for i in range(self.batch_size):
        random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
        #             random_str = '60/3=20'
        tmp = np.array(self.get_sequence_img(random_str))
        tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)
        tmp = tmp.transpose(1, 0, 2)

        X[i] = tmp
        y[i] = [self.characters.find(x) for x in random_str]

      i = 0
      XX = None
      yy = None
      for batch in self.datagen.flow(X, y, batch_size=self.batch_size):
        if not type(XX) == np.ndarray:
          XX = batch[0]
          yy = batch[1]
        else:
          XX = np.concatenate([XX, batch[0]], axis=0)
          yy = np.concatenate([yy, batch[1]], axis=0)

        i += 1
        if i >= self.gene:
          break
      yield [XX, yy, np.ones(self.batch_size * self.gene) * self.rnn_step,
             np.ones(self.batch_size * self.gene) * self.n_len], \
            np.ones(self.batch_size * self.gene)


if __name__ == '__main__':
  gen = GenSequenceChars('/Users/imperatore/data/letters',
                         10,
                         (280, 28, 1),
                         10,
                         rnn_step=128, gene=1)
  data = gen.gen()
  print(data)
  i = 0
  for d in data:
    [X, y, _, _], _ = d
    plt.imshow(X[0, :, :, 0])
    plt.title(','.join(map(lambda i: str(i), y[0])))
    plt.show()
    i += 1
    if i > 0:
      sys.exit(0)

# coding: utf-8

import random
import cv2
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import tensorflow as tf

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIGITS = '0123456789'
# characters = '0123456789+-*/=()'
characters = '0123456789'
width, height, n_len, n_class = 280, 28, 8, len(characters) + 1

datagen = ImageDataGenerator(
  rotation_range=0.4,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.2,
  zoom_range=0.2,
  fill_mode='nearest')


def plot(img, title):
  print('打印')
  plt.imshow(img[:, :, 0])
  plt.title(title)
  plt.show()


def get_img_by_char(char, base_path='../../dataset/nums'):
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
  return x


def gen(batch_size=128, gene=1, time_steps=0):
  """
  Generate batch data for training and test.
  Args:
    batch_size: The size of a batch(default 128).
    gene: Number of gene of a image.
  Return:
    arg0: A list contains data, label, rnn time steps, label sequence length.
    arg1: A useless 1D data(The fit_generator's data generator must contain X and y, but we use ctc_loss as a layer.
          so the loss calculation in this layer.
  """
  X = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
  y = np.zeros((batch_size, n_len), dtype=np.uint8)
  while True:
    for i in range(batch_size):
      random_str = ''.join([random.choice(characters) for j in range(n_len)])

      # Get dy-length sequence's image.
      tmp = np.array(get_sequence_img(random_str))
      tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)
      tmp = tmp.transpose(1, 0, 2)

      shape = tmp.shape
      # print(X[i, 0: shape[0], 0: shape[1], :].shape, shape[0], shape[1])
      X[i, 0: shape[0], 0: shape[1], :] = tmp
      y[i, 0: len(random_str)] = [characters.find(x) for x in random_str]
    # plot(X[0], y[0])
    i = 0
    XX = None
    yy = None
    for batch in datagen.flow(X, y, batch_size=batch_size):
      #             print(batch[0].shape, batch[1].shape)

      if not type(XX) == np.ndarray:
        XX = batch[0]
        yy = batch[1]
      else:
        XX = np.concatenate([XX, batch[0]], axis=0)
        yy = np.concatenate([yy, batch[1]], axis=0)

      i += 1
      if i >= gene:
        break
    yield [XX, yy, np.ones(batch_size * gene) * time_steps, np.ones(batch_size * gene) * n_len], \
          np.ones(batch_size * gene)
    # print('input shape', np.ones(batch_size * gene) * time_steps)
    # yield [XX, yy, np.array([1, 2]), np.array([1, 2])], \
    #       np.ones(batch_size * gene)


"""
#################################################################
CRNN network class:
  The code below is CRNN network structure.
#################################################################
"""


class CRNN(object):
  """

  """

  def __init__(self, num_class, rnn_units, batch_size):
    self.num_class = num_class
    self.rnn_units = rnn_units
    self.batch_size = batch_size

  def make_parallel(self, model, gpu_count):
    """

    :param model:
    :param gpu_count:
    :return:
    """

    def get_slice(data, idx, parts):
      shape = tf.shape(data)
      size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
      stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
      start = stride * idx
      return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
      outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('tower_%d' % i) as scope:

          inputs = []
          # Slice each input into a piece for processing on this GPU
          for x in model.inputs:
            input_shape = tuple(x.get_shape().as_list())[1:]
            slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
            inputs.append(slice_n)

          outputs = model(inputs)

          if not isinstance(outputs, list):
            outputs = [outputs]

          # Save all the outputs for merging back together later
          for l in range(len(outputs)):
            outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
      merged = []
      for outputs in outputs_all:
        merged.append(Concatenate(axis=0)(outputs))

      return Model(model.inputs, merged)

  def ctc_lambda_func(self, args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]  # [batch, rnn-step(height), num_class]
    # input_length表示输入的序列长度，每条输入数据长度可能不一
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

  def _build_network(self):
    input_tensor = Input((width, height, 1))
    x = input_tensor

    for i in range(3):
      x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal', padding='SAME')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal', padding='SAME')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = MaxPooling2D(pool_size=(2, 2))(x)

    conv_shape = x.get_shape().as_list()
    self.time_steps = conv_shape[1]
    rnn_dimen = conv_shape[2] * conv_shape[3]
    # print(conv_shape, rnn_dimen)
    x = Reshape(target_shape=(self.time_steps, rnn_dimen))(x)  # [batch, width, height * units]

    x = Dense(self.rnn_units, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    gru_1 = GRU(self.rnn_units, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(self.rnn_units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])  # [batch, height, units]

    gru_2 = GRU(self.rnn_units, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(self.rnn_units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                 name='gru2_b')(gru1_merged)

    x = concatenate([gru_2, gru_2b])  # [batch, height, units * 2]
    x = Dropout(0.25)(x)
    # [batch, height, n_class]
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    self.base_model = Model(input=input_tensor, output=x)  # [batch, height, units * 2]

    # base_model2 = make_parallel(base_model, 4)

    labels = Input(name='the_labels', shape=[n_len], dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(self.ctc_lambda_func, name='ctc')([self.base_model.output, labels, input_length, label_length])
    print('loss out', loss_out)

    # Define The ultimate model.
    # inputs is the first return item of fit's data function's.
    # The second return item of fit's data function will be used in compile function.
    self.model = Model(inputs=(input_tensor, labels, input_length, label_length), outputs=loss_out)
    self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

  def train(self):
    self.early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    self.history = self.model.fit_generator(gen(self.batch_size, gene=8, time_steps=self.time_steps), steps_per_epoch=100,
                                            epochs=20,
                                            callbacks=[Evaluator(self.base_model), self.early_stopping],
                                            validation_data=gen(self.batch_size, gene=8, time_steps=self.time_steps), validation_steps=20)

  def save_model(self):
    self.base_model.save('./crnn_model_10.h5')
    print('save model done!')


class Evaluator(Callback):
  def __init__(self, base_model):
    self.accs = []
    self.base_model = base_model

  def on_epoch_end(self, epoch, logs=None):
    acc = self.evaluate(steps=20) * 100
    self.accs.append(acc)
    if acc > 80:
      self.base_model.save('./crnn_epoch.h5')
      print('Model saved!')
    print('acc: %f%%' % acc)

  def evaluate(self, batch_size=128, steps=10):
    batch_acc = 0
    generator = gen(batch_size)
    for i in range(steps):
      [X_test, y_test, _, _], _ = next(generator)
      y_pred = self.base_model.predict(X_test)
      shape = y_pred[:, :, :].shape
      ctc_decode = K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
      out = K.get_value(ctc_decode)[:, :n_len]
      print(y_test[0], out[0])
      if out.shape[1] == n_len:
        batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps


if __name__ == '__main__':
  crnn = CRNN(num_class=n_class, rnn_units=128, batch_size=32)
  crnn._build_network()
  crnn.train()
  # next(gen(1))

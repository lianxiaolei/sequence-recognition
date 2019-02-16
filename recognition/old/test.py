# coding:utf8


class CRNN():
  """

  """

  def __init__(self):
    kernel_shape = [3, 3, 1, 128]
    self.w00 = kernel_shape.copy()
    for i in range(3):
      for j in range(2):
        if i == 0 and j == 0:
          print('w%s%s' % (i, j), kernel_shape, self.__dict__)
          continue
        auto_change_channel = True if j < 1 else False
        kernel_shape = self._change_size(kernel_shape, auto_change_channel=auto_change_channel)

        # print('before evaluate', self.__dict__['w%s%s' % (i, j)])
        self.__dict__['w%s%s' % (i, j)] = kernel_shape.copy()
        print('w%s%s' % (i, j), kernel_shape, self.__dict__)
        # print('w%s%s' % (i, j), self.__dict__['w%s%s' % (i, j)])
        # print()

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


if __name__ == '__main__':
  cr = CRNN()
  print(cr.__dict__)

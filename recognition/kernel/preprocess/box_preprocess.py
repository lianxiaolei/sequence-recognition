# coding:utf8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def plot(img, title=''):
  plt.imshow(img)
  plt.title(title)
  plt.show()


def rm_background(img, ):
  mask = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)

  kernel = np.ones((5, 5), np.uint8)

  img_mask = img * (1 - mask)

  img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
  img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)
  img_mask[img_mask > 0] = 1

  img = img * img_mask

  return img

def preprocess(gray):
  """
  :param img:
  :return:
  """
  sobel = cv2.Sobel(gray, cv2.CV_8U, dx=1, dy=0, ksize=3)
  # plt.imshow(sobel)
  # plt.show()

  ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU)

  print(ret)
  plt.imshow(binary)
  plt.title('ori binary')
  plt.show()

  element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 6))
  element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 4))
  #
  # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 18))
  # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (72, 12))

  dilation = cv2.dilate(binary, element2, iterations=1)
  plt.imshow(dilation)
  plt.title('dil0')
  plt.show()

  erosion = cv2.erode(dilation, element1, iterations=1)
  plt.imshow(erosion)
  plt.title('ero0')
  plt.show()

  median = cv2.medianBlur(erosion, 33)
  plt.imshow(median)
  plt.title('median')
  plt.show()

  # mask = 255 - cv2.adaptiveThreshold(
  #     sobel, 255,
  #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
  #     3, 3)
  # plt.imshow(mask)
  # plt.title('mask')
  # plt.show()

  dilation1 = cv2.dilate(median, element2, iterations=2)
  plt.imshow(dilation1)
  plt.title('dil1')
  plt.show()

  erosion1 = cv2.erode(dilation1, element2, iterations=1)
  plt.imshow(erosion1)
  plt.title('ero1')
  plt.show()

  dilation2 = cv2.dilate(erosion1, element2, iterations=2)
  plt.imshow(dilation2)
  plt.title('dil2')
  plt.show()

  return dilation2


if __name__ == '__main__':
  img = cv2.imread("../../../dataset/images/bk.jpg", cv2.IMREAD_GRAYSCALE)
  img = 255 - img
  img1 = img / 255.0
  rm_bk = rm_background(img)

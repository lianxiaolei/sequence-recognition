# coding: utf-8

import tensorflow as tf
import numpy as np
import time


class CNN(object):
    """

    """

    def __init__(self, p_keep_conv=1.0, p_keep_hidden=1.0,
                 batch_size=512, test_size=256, epoch_time=3):
        """
        Initialization
        :param batch_size
        :param test_size
        """
        self.w = self._init_weights([3, 3, 1, 32])  # 第一层卷积核大小为3x3,输入一张图,输出32个feature map
        self.w2 = self._init_weights([3, 3, 32, 64])  # 第二层卷积核大小为3x3,输入32个feature map,输出64个feature map
        self.w3 = self._init_weights([3, 3, 64, 148])  # 第三层卷积核大小为3x3,输入64个feature map,输出128个feature map
        self.w4 = self._init_weights([148 * 6 * 6, 1000])
        self.w_o = self._init_weights([1000, 10])  # FC 512 inputs, 10 outputs (labels)
        self.batch_size = batch_size
        self.test_size = test_size
        self.sess = tf.Session()  # tensorflow session
        self.p_keep_conv = p_keep_conv
        self.p_keep_hidden = p_keep_hidden
        self.epoch_time = epoch_time

    def _init_weights(self, shape, name=None):
        """
        初始化参数
        :param shape:
        :param name:
        :return:
        """
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def _cnn(self, X, p_keep_conv, p_keep_hidden):
        """
        :param X:
        :param p_keep_conv:
        :param p_keep_hidden:
        :return:
        """
        # 第一个卷积层:padding=SAME,保证输出的feature map与输入矩阵的大小相同
        l1a = tf.nn.relu(tf.nn.conv2d(X, self.w,  # l1a shape=(?, 48, 48, 32)
                                      strides=[1, 1, 1, 1], padding='SAME'))

        # max_pooling,窗口大小为2x2
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 24, 24, 32)
                            strides=[1, 2, 2, 1], padding='SAME')
        # dropout:每个神经元有p_keep_conv的概率以1/p_keep_conv的比例进行归一化,有(1-p_keep_conv)的概率置为0
        l1 = tf.nn.dropout(l1, p_keep_conv)

        # 第二个卷积层
        l2a = tf.nn.relu(tf.nn.conv2d(l1, self.w2,  # l2a shape=(?, 24, 24, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 12, 12, 64)
                            strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, p_keep_conv)

        # 第三个卷积层
        l3a = tf.nn.relu(tf.nn.conv2d(l2, self.w3,  # l3a shape=(?, 12, 12, 128)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 6, 6, 128)
                            strides=[1, 2, 2, 1], padding='SAME')
        # 将所有的feature map合并成一个2048维向量
        l4 = tf.reshape(l3, [-1, self.w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
        l4 = tf.nn.dropout(l4, p_keep_conv)
        # 后面两层为全连接层
        l4 = tf.nn.relu(tf.matmul(l4, self.w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)
        out = tf.matmul(l4, self.w_o)

        return out, l3

    def _cnn_new(self, X, p_keep_conv, p_keep_hidden):
        """
        :param X:
        :param p_keep_conv:
        :param p_keep_hidden:
        :return:
        """
        # 第一个卷积层:padding=SAME,保证输出的feature map与输入矩阵的大小相同
        l1 = tf.nn.relu(tf.nn.conv2d(X, self.w, strides=[1, 1, 1, 1], padding='SAME'))

        # max_pooling,窗口大小为2x2
        l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        l1 = tf.nn.dropout(l1, p_keep_conv)

        # 第二个卷积层
        l2 = tf.nn.relu(tf.nn.conv2d(l1, self.w2, strides=[1, 1, 1, 1], padding='SAME'))

        l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        l2 = tf.nn.dropout(l2, p_keep_conv)

        # 第三个卷积层
        l3 = tf.nn.relu(tf.nn.conv2d(l2, self.w3, strides=[1, 1, 1, 1], padding='SAME'))

        l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1],  # l3 shape=(?, 6, 6, 128)
                            strides=[1, 2, 2, 1], padding='SAME')

        # 将所有的feature map合并成一个2048维向量
        l4 = tf.reshape(l3, [-1, self.w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
        l4 = tf.nn.dropout(l4, p_keep_conv)
        # 后面两层为全连接层
        l4 = tf.nn.relu(tf.matmul(l4, self.w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)
        out = tf.matmul(l4, self.w_o)
        # out = tf.nn.softmax(out)

        return out, l3

    def fit_new(self, train_x, train_y, test_x, test_y):
        """
        :return:
        """
        p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
        p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
        X = tf.placeholder("float", [None, 48, 48, 1])
        Y = tf.placeholder("float", [None, 10])

        out, _ = self._cnn_new(X, p_keep_conv, p_keep_hidden)

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=out, labels=Y))  # 交叉熵目标函数

        train_op = tf.train.AdamOptimizer(0.001, 0.9) \
            .minimize(cost)  # RMSPro算法最小化目标函数

        predict_op = tf.argmax(out, 1)  # 返回每个样本的预测结果

        init = tf.global_variables_initializer()

        self.sess.run(init)
        for i in range(self.epoch_time):
            training_batch = zip(range(0, len(train_x), self.batch_size),
                                 range(self.batch_size, len(train_x) + 1, self.batch_size))
            tm = time.time()
            for start, end in training_batch:
                # print(train_x[start:end].shape, train_y[start:end].shape)
                self.sess.run(
                    train_op,
                    feed_dict={X: train_x[start:end],
                               Y: train_y[start:end],
                               p_keep_conv: self.p_keep_conv,
                               p_keep_hidden: self.p_keep_hidden})

            test_indices = np.arange(len(test_x))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0: self.test_size]

            acc = np.mean(np.argmax(test_y[test_indices], axis=1) ==
                                          self.sess.run(
                                              predict_op, feed_dict={
                                                  X: test_x[test_indices],
                                                  p_keep_conv: 1.0,
                                                  p_keep_hidden: 1.0}))

            print('epoch %s' % (i + 1), 'accuracy %s' % acc, '\t time cost %s' % (time.time() - tm))
            # if acc >= 1.0: break  # 提前终止

    def save(self, model_name):
        """

        :param model_name:
        :return:
        """
        saver = tf.train.Saver()
        saver.save(self.sess, model_name)
        print("save model:{0} Finished".format(model_name))

    def predict(self, img):
        """

        :param img:
        :return:
        """
        p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
        p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
        X = tf.placeholder("float", [None, 48, 48, 1])

        out = self._cnn_main(X, p_keep_conv, p_keep_hidden)
        predict_op = tf.argmax(out, 1)  # 返回每个样本的预测结果

        predict = self.sess.run(
            predict_op, feed_dict={
                X: img.reshape(-1, 48, 48, 1),
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})

        return predict

    def predict_new(self, img):
        """

        :param img:
        :return:
        """
        p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
        p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
        X = tf.placeholder("float", [None, 48, 48, 1])
        out, pool = self._cnn(X, p_keep_conv, p_keep_hidden)
        predict_op = tf.argmax(out, 1)  # 返回每个样本的预测结果
        predict = self.sess.run(
            predict_op, feed_dict={
                X: img.reshape(-1, 48, 48, 1),
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})
        return predict

    def output(self, img):
        """

        :param img:
        :return:
        """
        p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
        p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
        X = tf.placeholder("float", [None, 48, 48, 1])
        out, pool = self._cnn(X, p_keep_conv, p_keep_hidden)
        output = self.sess.run(
            pool, feed_dict={
                X: img.reshape(-1, 48, 48, 1),
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})
        probs = self.sess.run(
            out, feed_dict={
                X: img.reshape(-1, 48, 48, 1),
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})
        return output, probs

    def load_session(self, model_name):
        """

        :param model_name:
        :return:
        """
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, model_name)

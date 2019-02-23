# coding: utf-8

import cv2
from keras.callbacks import *
from keras.layers import *
from keras.models import *
import keras.backend as kb
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import tensorflow as tf
tf.keras.models.Model()

def train_generation():
    datagen = image.ImageDataGenerator(featurewise_center=False,
                                       samplewise_center=False,
                                       featurewise_std_normalization=False,
                                       samplewise_std_normalization=False,
                                       zca_whitening=False,
                                       rotation_range=0.3,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.,
                                       zoom_range=0.2,
                                       channel_shift_range=0.,
                                       fill_mode='nearest',
                                       cval=0.0,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       rescale=1. / 255,
                                       preprocessing_function=None,
                                       # data_format=K.image_data_format(),
                                       )

    train_generator = datagen.flow_from_directory(
        # '/Users/imperatore/tmp/num_ocr',  # this is the target directory
        r'F:\number_ok1',  # this is the target directory
        target_size=(48, 48),  # all images will be resized to 48*40
        batch_size=256,
        class_mode='categorical',
        color_mode='grayscale')

    validation_generator = datagen.flow_from_directory(
        # '/Users/imperatore/tmp/nums_classed',
        r'F:\number_ok1',
        target_size=(48, 48),
        batch_size=128,
        class_mode='categorical',
        color_mode='grayscale')

    num_class = 10
    input_tensor = Input((48, 48, 1))

    def resnet(input_tensor, units=32, kernel_size=(3, 3)):
        x = input_tensor
        for i in range(3):
            x = res_block(x, units, kernel_size=kernel_size)
            x = Dropout(drop)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        return x

    def conv2d_bn(x, units, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(units, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def res_block(inpt, units, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = conv2d_bn(inpt, units=units, kernel_size=kernel_size, strides=strides, padding='same')
        x = conv2d_bn(x, units=units, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = conv2d_bn(inpt, units=units, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    drop = 0.1
    input_tensor = Input((48, 48, 1))
    x = input_tensor

    x = Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), name=None)(x)
    x = Activation(activation='relu')(x)
    # x = BatchNormalization(axis=3, name=None)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(drop)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), name=None)(x)
    x = Activation(activation='relu')(x)
    # x = BatchNormalization(axis=3, name=None)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(drop)(x)

    # x = Conv2D(256, kernel_size=(3, 3), padding='same', strides=(1, 1), name=None)(x)
    # x = Activation(activation='relu')(x)
    # # x = BatchNormalization(axis=3, name=None)(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(drop)(x)

    x = Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1), name=None)(x)
    x = Activation(activation='relu')(x)
    # x = BatchNormalization(axis=3, name=None)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(drop)(x)

    # x = resnet(x)

    x = Flatten()(x)
    x = Dense(1000, kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(10, kernel_initializer='he_normal')(x)

    model = Model(inputs=input_tensor, outputs=x)
    print(model.layers)

    kb.set_learning_phase(0)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print('\n'.join([str(tmp) for tmp in model.layers]))
    print('model length: %s' % len(model.layers))

    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    model.fit_generator(
        train_generator,
        steps_per_epoch=512,
        epochs=20,
        validation_data=validation_generator,
        nb_val_samples=100,
        verbose=True,
        callbacks=[early_stopping])

    model.save('cnn3_gen_1.4.h5')  # always save your weights after training or during training


def predict_generation():
    model = load_model('cnn3_gen_1.1_ftn.h5')

    datagen = image.ImageDataGenerator(featurewise_center=False,
                                       samplewise_center=False,
                                       featurewise_std_normalization=False,
                                       samplewise_std_normalization=False,
                                       zca_whitening=False,
                                       rotation_range=0.3,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.,
                                       zoom_range=0.2,
                                       channel_shift_range=0.,
                                       fill_mode='nearest',
                                       cval=0.0,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       rescale=1. / 255,
                                       preprocessing_function=None,
                                       # data_format=K.image_data_format(),
                                       )

    train_generator = datagen.flow_from_directory(
        # '/Users/imperatore/tmp/num_ocr',  # this is the target directory
        r'F:\234\character',  # this is the target directory
        target_size=(48, 48),  # all images will be resized to 48*40
        batch_size=256,
        class_mode='categorical',
        color_mode='grayscale')

    validation_generator = datagen.flow_from_directory(
        # '/Users/imperatore/tmp/nums_classed',
        r'F:\234\character',
        target_size=(48, 48),
        batch_size=256,
        class_mode='categorical',
        color_mode='grayscale')

    prediction = model.predict_generator(train_generator, len(train_generator))


if __name__ == '__main__':
    train_generation()
    from keras import backend as K
    K.function()

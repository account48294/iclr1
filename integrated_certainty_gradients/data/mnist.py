"""
    The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of
    images of numerals 0 to 9 in 28 x 28 pixels greyscale (0.0 to 1.0).
"""

import tensorflow as tf
from tensorflow import keras

from . import dataset


def mnist_data() -> dataset.Dataset:
    """
        :return: The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of
            images of numerals 0 to 9 in 28 x 28 pixels greyscale (0.0 to 1.0).
    """
    number_of_classes = 10

    def transform(x, y):
        return (
            tf.cast(tf.expand_dims(x, -1), tf.float32) / 255.,
            tf.one_hot(y, number_of_classes)
        )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return (
        dataset.Dataset.from_tensors(x_train, y_train, x_test, y_test)
        .map(transform)
    )


def mnist_in_space(right_align: bool = False) -> dataset.Dataset:
    """
        :return: Numerals from the MNIST dataset towards the left side of a
            larger (32x64) black containing image.
    """
    if right_align:
        left_pad = 33
        right_pad = 3
    else:
        left_pad = 3
        right_pad = 33

    def transform(image, label):
        # Augment with small shifts range -2 to 2
        shift_horizontal = tf.random.uniform((), -2, 3, tf.int32)
        shift_vertical = tf.random.uniform((), -2, 3, tf.int32)
        return (
            tf.pad(
                image,
                (
                    (2+shift_vertical, 2-shift_vertical),
                    (left_pad+shift_horizontal, right_pad-shift_horizontal),
                    (0, 0))),
            label
        )

    return mnist_data().map(transform)

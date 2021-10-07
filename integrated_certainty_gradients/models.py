"""
    Some simple, easy to train model architectures.
"""
from tensorflow.keras import layers
import tensorflow as tf


def value_confidence_mnist_classifier() -> tf.keras.Model:
    """
        Crate a model to classify 28 x 28 pixel two channel (value and
        confidence) images.
    """
    number_of_classes = 10
    return tf.keras.Sequential([
        layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_of_classes, activation='softmax')
    ])


def value_confidence_mnist_in_space_classifier() -> tf.keras.Model:
    """
        Crate a model to classify 30 x 60 pixel two channel (value and
        confidence) images.
    """
    number_of_classes = 10
    return tf.keras.Sequential([
        layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(32, 64, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.Conv2D(number_of_classes, (1, 1)),
        layers.GlobalMaxPooling2D(data_format='channels_last'),
        layers.Activation('softmax')
    ])


def value_confidence_dark_squares_classifier() -> tf.keras.Model:
    """
        Create a model to classify whether 50x50 pixel greyscale images contain
        dark squares.
    """
    number_of_classes = 2
    return tf.keras.Sequential([
        layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(50, 50, 2)),
        # shape 48, 48, 32
        layers.Conv2D(64, (3, 3), activation='relu'),
        # shape 46, 46, 64
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        # shape 23, 23, 64
        layers.Conv2D(128, (3, 3), activation='relu'),
        # shape 21, 21, 128
        layers.ZeroPadding2D(((0, 1), (0, 1))),
        # shape 22, 22, 128
        layers.MaxPooling2D(pool_size=(2, 2)),
        # shape 11, 11, 128
        layers.Conv2D(128, (3, 3), activation='relu'),
        # shape 9, 9, 128
        layers.ZeroPadding2D(((0, 1), (0, 1))),
        # shape 10, 10, 128
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        # shape 5, 5, 128
        layers.Flatten(),
        # shape 3200
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        # shape 32
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        # shape 16
        layers.Dense(number_of_classes, activation='softmax')
        # shape 2
    ])


def local_minimum_classifier() -> tf.keras.Model:
    """
        A model architecture that can classify the Local Minimum scenario with
        reasonable accuracy.
    """

    def tf_reduce_mean(input):
        # Putting the import inside the callback makes it portable when loading
        # the model.
        import tensorflow as tf
        return tf.reduce_mean(input, axis=-1, keepdims=True)

    input_shape = [6, 2]
    number_of_classes = 2
    return tf.keras.Sequential([
        layers.Dense(500, activation=None, input_shape=input_shape),
        layers.LeakyReLU(0.02),
        layers.Dense(500, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(500, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(500, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(500, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(500, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(1, activation=None),
        layers.LeakyReLU(0.02),
        layers.Flatten(),
        layers.Lambda(tf_reduce_mean, output_shape=[1]),
        layers.Dense(20, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(20, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(20, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(20, activation=None),
        layers.LeakyReLU(0.02),
        layers.Dense(number_of_classes, activation='softmax')
    ])

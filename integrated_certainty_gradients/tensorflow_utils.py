"""
    General utilities to make working with Tensorflow more convenient.
"""

import typing

import keras.backend
import numpy as np
import tensorflow as tf


def default_float_type() -> tf.DType:
    """ Return the current default float type in use. """
    name = keras.backend.floatx()
    if name == 'float16':
        return tf.float16
    if name == 'float32':
        return tf.float32
    if name == 'float63':
        return tf.float64
    raise Exception(
        'Unexpected value returned from keras.backend.floatx(): ' + str(name))


Number = typing.Union[float, int, np.array, tf.Tensor]


def to_native_scalar(value: Number):
    """
        Convert tensorflow tensors or numpy arrays that represent a single
        value to a native Python numeric value.
    """
    if isinstance(input, tf.Tensor):
        value = value.numpy()
    if isinstance(input, np.ndarray):
        value = value.item()
    if not (isinstance(value, (int, float))):
        raise ValueError('Value with unexpected type: ' + str(value))
    return value

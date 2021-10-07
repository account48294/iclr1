"""
    Diagnostic tools for debugging and introspecting Tensorflow evaluation.
"""


import sys
import typing

import tensorflow as tf


def print_pipe(
        message: str = None, shape: bool = True, value: bool = False
        ) -> typing.Any:
    """
        Create a function that can wrap a tensor so that when it is evaluated
        diagnostic information is printed out.

        :param message: Print a message.
        :param shape: Print the shape of the tensor.
        :param value: Print the value of the tensor.
        :return: A callback to wrap a lazy Tensor.
    """

    @tf.function
    def result(input_tensor):
        tf.print('', output_stream=sys.stdout)
        if message is not None:
            tf.print(message, output_stream=sys.stdout)
        if shape:
            tf.print(['shape:', input_tensor.shape], output_stream=sys.stdout)
        if value:
            tf.print('value:', output_stream=sys.stdout)
            tf.print(input_tensor, output_stream=sys.stdout)
        return input_tensor

    return result

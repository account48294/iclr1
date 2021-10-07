"""
    Provides data for the Local Minimum scenario. This scenario challenges a
    model to detect whether the "scores" of elements in a vector sum to above
    0 or not. The "scores" are given by a simple deterministic scalar function.
    The function has a positive local minimum. This is intended to be
    pathological for the Additive Gaussian Noise attribution method.
"""

import typing

import tensorflow as tf

from . import dataset
from .. import pixel_certainty


def scores(input_example: tf.Tensor) -> tf.Tensor:
    """
        Generate the per-element "scores" (as defined by the Local Minimum
        scenario) for an input tensor.
    """
    return tf.where(
        input_example > 0.75,
        2 * input_example - 1.,
        tf.where(
            input_example > 0.5,
            2 * (1. - input_example),
            -0.75
        )
    )


def local_minimum_dataset() -> dataset.Dataset:
    """
        Generate a dataset for the Local Minimum scenario.

        Both the train and evaluation data are autogenerated. To create static
        data use the example method in this module.
    """

    input_shape = [6, 1]

    def generator_dataset(size):

        def generate_sample():
            for _ in range(size):
                covariate = tf.random.uniform(input_shape, 0., 1.)
                class_number = tf.cast(
                    tf.reduce_sum(scores(covariate)) > 0., tf.uint8)

                yield covariate, tf.one_hot(class_number, 2, 1., 0.)

        return tf.data.Dataset.from_generator(
            generate_sample,
            output_signature=(
                tf.TensorSpec(input_shape, tf.float32),
                tf.TensorSpec([2], tf.float32)
            ))

    return dataset.Dataset(generator_dataset(100000), generator_dataset(500))


def example(
        data: typing.Iterable = (0.1, 0.2, 0.3, 0.6, 0.7, 0.9),
        certainty_channel: bool = True, batched: bool = True,
        as_image: bool = False) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
        Create an example input (by default with values .1, .2, .3, .6, .7, .9)

        :param data: The values to put in the vector.
        :param certainty_channel: Should a certainty channel be added. If it
            is, the certainty values will all be 1.
        :param batched: Should a size 1 batch axis be prepended.
        :param as_image: Should an additional size 1 axis be prepended to make
            the result a 1 pixel high image.
        :return: The data as a tensor ready for input to a model.
    """
    data = tf.convert_to_tensor(list(data))
    if certainty_channel:
        data = pixel_certainty.add_certainty_channel(
            tf.expand_dims(data, axis=-1))
    if as_image:
        data = tf.expand_dims(data, axis=0)
    label = tf.one_hot([0], 2, 1., 0.)
    if batched:
        data = tf.expand_dims(data, axis=0)
        label = tf.expand_dims(label, axis=0)
    return data, label

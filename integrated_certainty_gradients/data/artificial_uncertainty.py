"""
    Tools for creating certainty-aware and certainty training datasets.
"""

import math
import typing

import scipy.optimize

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfp_distributions

from . import dataset
from .. import pixel_certainty


def with_certainty(data: dataset.Dataset) -> dataset.Dataset:
    """
        :return: The input dataset with an additional channel added to the last
        axis with value 1.0
    """
    return data.map_x(pixel_certainty.add_certainty_channel)


def _continuous_bernoulli_parameter_for_mean(mean):
    """
        This function takes a value and returns the parameter of the
        continuous bernoulli function that will yield that mean. It is the
        inverse of the parameter to distribution mean function.
    """

    def expectation(parameter):
        if parameter == 1.:
            return 1.
        if parameter == 0.5:
            return 0.5
        if parameter == 0.:
            return 0.
        return parameter / (2 * parameter - 1) + 1. / (
            2 * math.atanh(1 - 2 * parameter))

    result = scipy.optimize.root_scalar(
        lambda parameter: expectation(parameter) - mean,
        bracket=(0., 1.),
        xtol=1.e-16)
    if not result.converged:
        raise Exception('Failed to calculate inverse expectation.')
    return result.root


def constant_mean_continuous_bernoulli(
        mean: float, shape: typing.Tuple) -> tfp_distributions.Distribution:
    """
        Return values from the Continuous Bernoulli distribution as defined in
        https://arxiv.org/pdf/1907.06845.pdf

        This method is faster than mean_continuous_bernoulli and should be
        preferred when all samples have the same mean.
    """
    parameter = _continuous_bernoulli_parameter_for_mean(mean)
    return tfp_distributions.ContinuousBernoulli(
        probs=tf.fill(shape, parameter))


def mean_continuous_bernoulli(
        means: tf.Tensor) -> tfp_distributions.Distribution:
    """
        Return values from the Continuous Bernoulli distribution as defined in
        https://arxiv.org/pdf/1907.06845.pdf

        Takes only scalar mean values because a closed form formula for one
        required calculation is not known, and a time consuming approximation
        must be used during processing of the mean.
    """
    original_shape = tf.shape(means)
    dimensions_count = tf.reshape(tf.reduce_prod(original_shape), [1])
    flat_means = tf.reshape(means, dimensions_count)

    def numpy_function(numpy_array):
        result = np.array(
            _continuous_bernoulli_parameter_for_mean(numpy_array.item()),
            np.float64)
        return result

    def tensorflow_function(mean):
        result = tf.numpy_function(
            numpy_function, [tf.cast(mean, tf.float64)], tf.float64)
        return result

    flat_parameters = tf.map_fn(
        tensorflow_function, tf.cast(flat_means, tf.float64))
    parameters = tf.reshape(flat_parameters, original_shape)
    return tfp_distributions.ContinuousBernoulli(probs=parameters)


def damaged(
        data: typing.Union[dataset.Dataset, tf.Tensor],
        binary_damage: bool = True, adversarial_damage: bool = True,
        variable_extents: bool = False) -> dataset.Dataset:
    """
        The input with an additional "certainty" channel added and then
        "damaged" (see pixel_certainty.damage_certainty).

        :return: The "damaged" Dataset.
    """

    def damage_tensor(images):
        images_shape = tf.shape(images)
        if variable_extents:
            damage_extents = tf.random.uniform(
                images_shape[0:1], 0., 1., tf.float64)
        else:
            damage_extents = tf.fill(images_shape[0:1], 0.5)
        if binary_damage:
            distribution = tfp_distributions.Bernoulli(
                probs=damage_extents, dtype=tf.float32)
        else:
            distribution = mean_continuous_bernoulli(damage_extents)
        result = distribution.sample(images_shape[1:-1])
        # Bring the last axis to the front
        axes_count = len(images.shape) - 1
        transposition = [axes_count - 1] + list(range(0, axes_count - 1))
        return tf.transpose(result, transposition)

    def damage_tensors(images):
        if adversarial_damage:
            background = tf.random.shuffle(images)
        else:
            background = None

        return pixel_certainty.damage_certainty(
            pixel_certainty.add_certainty_channel(images),
            tf.expand_dims(damage_tensor(images), -1),
            background_tensor=background)

    if isinstance(data, tf.Tensor):
        return damage_tensors(data)
    else:
        return data.batch(200).map_x(damage_tensors).unbatch()


def baselines(data: dataset.Dataset) -> dataset.Dataset:
    """
        Returns a dataset where the pixel certainty is uniformly 0.0, the y
        values are equiprobable across all categories, and the pixel values
        are taken from the mnist dataset.

        This data assumes an equiprobable choice for unknown data is
        appropriate, rather than reflexting the distribution in the original
        dataset, if it is not balanced between all categories.

        :return: The "baseline" Dataset.
    """
    def transform_data(x, y):
        return (
            pixel_certainty.add_certainty_channel(x, 0.),
            tf.fill(y.shape, 1. / y.shape[-1])
        )
    return data.map(transform_data)


def mixed_damage(data: dataset.Dataset) -> dataset.Dataset:
    """
        Create a dataset using a mixture of damage methods.

        :param data: A dataset without a certainty channel.
        :return: A dataset with an added certainty channel and damage applied.
    """
    return dataset.combine([
        damaged(
            data, binary_damage=True, adversarial_damage=True,
            variable_extents=True),
        damaged(
            data, binary_damage=False, adversarial_damage=False,
            variable_extents=True),
        damaged(
            data, binary_damage=True, adversarial_damage=False,
            variable_extents=True),
        damaged(
            data, binary_damage=False, adversarial_damage=True,
            variable_extents=True)
    ])

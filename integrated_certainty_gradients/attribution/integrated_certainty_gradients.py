"""
    This module provides attribution methods based on Integrated Gradient using
    Artificial Uncertainty baselines.
"""

import typing

import keras
import tensorflow as tf
import numpy as np

from . import integrated_gradients
from .. import pixel_certainty, tensor_tools
from ..data import artificial_uncertainty


def certainty_aware_simple_integrated_gradients(
        images: tf.Tensor, model: keras.Model,
        baseline_value: typing.Union[float, tf.Tensor],
        output_class: int = None, display_partial_sums: bool = False
        ) -> tf.Tensor:
    """
        Perform an integrated gradients attribution where the baseline is
        filled with a single constant value, but the certainty channel is
        excluded from the calculation.
    """
    if (len(images.shape)) != 4:
        raise ValueError(
            'Image input should have shape (sample count, row count, '
            + 'column count, channel count)')
    value_channel_count = images.shape[-1] - 1
    baseline_value = tf.convert_to_tensor(baseline_value)
    if len(baseline_value.shape) == 0:
        baseline_value = baseline_value + tf.zeros([value_channel_count])
    elif len(baseline_value.shape) > 1:
        raise ValueError('Only a single channel axis is supported.')
    if baseline_value.shape[0] == 1:
        baseline_value = baseline_value * tf.ones(value_channel_count)
    elif baseline_value.shape[0] != value_channel_count:
        raise ValueError(
            'The baseline must be a single value or a 1d vector with size '
            + 'equal to the number of value channels (last axis size - 1)')
    masked_baseline = tf.concat([baseline_value, [0]], axis=0)
    masked_baseline = tf.reshape(
        masked_baseline, [1, 1, 1, value_channel_count + 1])
    image_mask = tf.concat([tf.zeros([value_channel_count]), [1]], axis=0)
    image_mask = tf.reshape(image_mask, [1, 1, 1, value_channel_count + 1])
    baseline = masked_baseline + (images * image_mask)
    return integrated_gradients.integrated_gradients(
        images, model, baseline, output_class=output_class,
        display_partial_sums=display_partial_sums)


def certainty_aware_double_sided_integrated_gradients(
        images: tf.Tensor, model: keras.Model, minimum: float = 0.,
        maximum: float = 1., output_class: int = None,
        display_partial_sums: bool = False) -> tf.Tensor:
    """
        Performs Integrated Gradients taking an average from the minimum
        (default 0.0) and maximum (default 1.0) baselines.
    """
    lower = certainty_aware_simple_integrated_gradients(
        images, model, minimum, output_class=output_class,
        display_partial_sums=display_partial_sums)
    upper = certainty_aware_simple_integrated_gradients(
        images, model, maximum, output_class=output_class,
        display_partial_sums=display_partial_sums)
    return (lower + upper) / 2.


def image_integrated_certainty_gradients(
        image: tf.Tensor, model: keras.Model, output_class: int = None,
        display_partial_sums: bool = False) -> tf.Tensor:
    """
        Calculate the salience of each pixel to the prediction the model makes
        for each image using the Integrated Uncertainty Gradients method.

        The Integrated Certainty Gradients method builds upon the Integrated
        Gradients method (integrated_gradients). Instead of taking a user
        supplied baseline, it operates on data with uncertainty semantics,
        which implies a canonical baseline of maximum uncertainty.

        :param image: The image for which the salience of input pixels to the
            prediction is desired to be known.
        :param model: The model which is making the predictions.
        :param output_class: The class for which the prediciton probability
            attributions are calculated. If None, the class predicted by the
            model for each image is used.
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """
    zero_confidence_vector = [np.float32(1.)] * image.shape[-1]
    zero_confidence_vector[-1] = np.float32(0.)
    baseline = image * zero_confidence_vector
    return integrated_gradients.integrated_gradients(
        image, model, baseline, output_class=output_class,
        display_partial_sums=display_partial_sums)


def image_expected_certainty_gradients(
        image: tf.Tensor, model: keras.Model, samples: int = 500,
        output_class: int = None, display_partial_sums: bool = False
        ) -> tf.Tensor:
    """
        Perform a hybrid Expected Gradients and Integrated Certainty Gradients
        attribution. Use randomly degraded certainties of the input image as
        a baseline.

        :param image: The input vector.
        :param model: The model making the prediction to attribute.
        :param samples: The number of samples to average over.
        :param output_class: Prediction class to calculate attribution for. If
            the argument is None, the predicted (highest value) class will be
            used.
    """
    if image.shape[0] == 1:
        raise ValueError(
            'Image argument of image_expected_certainty_gradients should not '
            + "have a 'samples' axis.")
    if len(image.shape) != 3:
        raise ValueError(
            'Image argument of image_expected_certainty_gradients should have '
            + "3 axes: 'row', 'column', 'channel'.")
    baselines_values_shape = tf.concat(
        [[samples], image.shape[0:2], [image.shape[2] - 1]], 0)
    baselines_certainty_shape = tf.concat(
        [[samples], image.shape[0:2], [1]], 0)
    baselines = image * tf.concat(
        [
            tf.ones(baselines_values_shape),
            #artificial_uncertainty.constant_mean_continuous_bernoulli(
            #    random.random(), baselines_certainty_shape).sample()
            tf.random.uniform(baselines_certainty_shape),
        ],
        -1)
    return integrated_gradients.integrated_gradients(
        image, model, baselines, True, output_class=output_class,
        display_partial_sums=display_partial_sums)


def random_path_certainty_gradients(
        images: tf.Tensor, model: keras.Model, samples: int = 100,
        output_class: int = None, display_partial_sums: bool = False
        ) -> tf.Tensor:
    """
        Perform integrated certainty gradients but starting at random certainty
        baselines instead of zero certainty.

        :param images: The images to attribute.
        :param model: The model to attribute.
        :param samples: How many random samples to average over.
        :param output_class: Which class to return attributions for. If None,
            the highest probability class will be used.
        :param display_partial_sums: Show on the screen images of the
            intermediate calculations.
        :return:
    """
    class_count = model.output_shape[-1]
    if output_class is not None:
        if output_class < 0:
            raise ValueError(
                'The output class index (' + str(output_class)
                + ') should be 0 or greater.')
        if output_class >= class_count:
            raise ValueError(
                'The output class index (' + str(output_class)
                + ') should be less than the number of output classes ('
                + str(class_count) + ').')
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
    # Axes: (image, certainty_sample, row, column, channel)
    image_count = images.shape[0]
    row_count = images.shape[1]
    column_count = images.shape[2]
    channel_count = images.shape[3]
    evaluations_shape = [image_count, samples]
    pixels_shape = [row_count, column_count]
    image_shape = pixels_shape + [channel_count]
    bernoulli_means = tf.random.uniform(evaluations_shape, 0., 1.)

    certainty_values = (
        artificial_uncertainty
        .mean_continuous_bernoulli(bernoulli_means)
        .sample(pixels_shape)
    )
    # Put image and certainty sample at start, not the end
    certainty_values_normal_order = tf.transpose(
        certainty_values, [2, 3, 0, 1])
    certainty_samples = tf.reshape(
        # tf.random.uniform(
        #    evaluations_shape + pixels_shape, 0., 1.),
        certainty_values_normal_order,
        evaluations_shape + pixels_shape + [1])
    certainty_mask_color_channels = tf.ones(
        evaluations_shape + pixels_shape + [channel_count - 1])
    certainty_masks = tf.concat(
        [certainty_mask_color_channels, certainty_samples], -1)
    interpolated_images = images * certainty_masks
    if output_class is None:
        predictions = tf.argmax(model(images), axis=-1)
    else:
        predictions = tf.fill([images.shape[0]], output_class)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        unstructured_shape = [np.prod(evaluations_shape)] + image_shape
        unstructured_outputs = model(tf.reshape(
            interpolated_images, unstructured_shape))
        structured_outputs = tf.reshape(
            unstructured_outputs, evaluations_shape + [class_count])
        # prediction_indices are the indices of the output values of interest
        # "3" for the two evaluations axes plus one for the coordinates
        prediction_indices = np.zeros(evaluations_shape + [3], np.int32)
        for coordinates in tensor_tools.Coordinates(evaluations_shape):
            prediction = predictions[coordinates[0]]
            prediction_indices[tuple(coordinates)] = coordinates + [prediction]
        prediction_outputs = tf.gather_nd(
            structured_outputs, prediction_indices)
        gradients = tape.gradient(
            prediction_outputs, interpolated_images)
    # "partial_gradients" in the sense of partial derivatives.
    partial_gradients = gradients[..., -1]
    if display_partial_sums:
        if image_count > 1:
            raise ValueError(
                'Cannot display interpolations for multi-image attribution.')
        integrated_gradients.display_interpolations(
            # Expand dims because we have only 1 interpolations axis but usual
            # integrated gradients uses 2.
            tf.expand_dims(interpolated_images, 1),
            tf.expand_dims(partial_gradients, 1))
    return tf.math.reduce_mean(partial_gradients, axis=[1])


def gaussian_certainty_gradients(
        image: tf.Tensor, model: keras.Model, standard_deviation: float = 0.1,
        minimum: float = 0., maximum: float = 1.,
        baseline_multiplicty: int = 100,
        output_class: typing.Optional[int] = None) -> tf.Tensor:
    """
        Perform Integrated Certainty Gradients with an Additive Gaussian Noise
        baseline. Use randomly degraded certainties of the input
        image as a baseline.

        :param image: The input vector.
        :param model: The model making the prediction to attribute.
        :param samples: The number of samples to average over.
        :param output_class: Prediction class to calculate attribution for. If
            the argument is None, the predicted (highest value) class will be
            used.
    :return:
    """
    baseline_valuess_shape = image.shape.as_list()
    baseline_valuess_shape[0] = baseline_multiplicty
    baseline_valuess_shape[-1] = baseline_valuess_shape[-1] - 1
    baseline_certainties_shape = baseline_valuess_shape.copy()
    baseline_certainties_shape[-1] = 1
    noise = tf.random.normal(baseline_valuess_shape, 0, standard_deviation)
    baseline_values = tf.maximum(minimum, tf.minimum(
        maximum, pixel_certainty.discard_certainty(image) + noise))
    baseline_certainties = tf.ones(baseline_certainties_shape)
    baselines = tf.concat([baseline_values, baseline_certainties], axis=-1)
    return integrated_gradients.integrated_gradients(
        image, model, baselines, interpolation_count=20,
        output_class=output_class)

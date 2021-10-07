"""
    Capabilities associated with the Integrated Gradients attribution method.
"""
import math
import typing
from typing import List

import keras
import numpy as np
import tensorflow as tf

from .. import image_tensors, tensor_tools


def classifier_gradients(
        inputs: tf.Tensor, model: keras.Model, class_names: List[str] = None
        ) -> tf.Tensor:
    """
        Calculates the gradients of a classifier prediction with respect to its
        inputs.

        The gradient of the probability of the chosen class with respect to the
        elements of the input vectors is calculated. It is assumed the first
        axis is the sample number and the remaining axes are of the input
        vectors.
        :param inputs: The inputs ("x vectors") to the model.
        :param model: The model classifying the inputs.
        :param class_names: The names associated
        :return: A tensor containing the gradients for each element of each
            input.
    """
    # TODO: Implement class_names
    if class_names is not None:
        raise NotImplementedError('class_names parameter is not implemented.')
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        logits = model(inputs)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicted_class_indices = tf.argmax(probabilities, axis=-1)
        predicted_class_indices_as_tensor_coordinates = []
        for index in range(inputs.shape[0]):
            predicted_class_indices_as_tensor_coordinates.append(
                [index, predicted_class_indices[index]])
        predicted_class_probabilities = tf.gather_nd(
            probabilities, predicted_class_indices_as_tensor_coordinates)
        return tape.gradient(predicted_class_probabilities, inputs)


def display_interpolations(
        interpolated_images: tf.Tensor, partial_gradients: tf.Tensor,
        certainty_channel: bool = False) -> None:
    """
        Display the intermediate calculated images from the Integrated
        Gradients method.

        :param interpolated_images: Interpolations between a baseline and
            target image.
        :param partial_gradients: Partial sums of gradients of interpolated
            images up to the current step.
        :param certainty_channel: Do the images contain a certainty channel
            (basic images do not).
    """

    certainty_channel = True

    max_displayed_interpolations_count = 10
    # If the displayed gradients are all very faint, it indicates that
    # the displayed interpolations do not include the most important ones
    # (ie. they are not representative).
    if interpolated_images.shape[:-1] != partial_gradients.shape:
        raise ValueError(
            'interpolated_images and partial_gradients shapes do not match.')
    # Remove the image axis: display can only be done for one image
    interpolated_images = tf.squeeze(interpolated_images, [0])
    partial_gradients = tf.squeeze(partial_gradients, [0])
    # Remove either baselines or interpolations: display cannot be done for
    # both.
    if interpolated_images.shape[0] == 1:
        interpolated_images = tf.squeeze(interpolated_images, [0])
        partial_gradients = tf.squeeze(partial_gradients, [0])
    else:
        interpolated_images = tf.squeeze(interpolated_images, [1])
        partial_gradients = tf.squeeze(partial_gradients, [1])
    if len(interpolated_images.shape) != 4:
        raise ValueError(
            'To display interpolations, images must have a single '
            + 'interpolation axis, two pixel axes and one optional color '
            + 'axis.')
    number_of_channels = interpolated_images.shape[-1]

    # TODO ALSO FOR MULTIPLE BASELINES (CHOOSE 1 MAYBE)

    if certainty_channel:
        if number_of_channels in [2, 4]:
            certainty_interpolations = interpolated_images[..., -1]
            interpolated_images = interpolated_images[..., :-1]
        else:
            raise ValueError(
                'To display interpolations with certainty, images must have '
                + 'one or three color channels and a single certainty '
                + 'channel.')
    elif number_of_channels not in [1, 3]:
        raise ValueError(
            'To display interpolations, images must have one or three color '
            + 'channels.')
    provided_interpolations_count = interpolated_images.shape[0]
    step_size = math.ceil(
        provided_interpolations_count / max_displayed_interpolations_count)
    displayed_interpolation_indices: typing.Iterable[int] = range(
        0, provided_interpolations_count, step_size)
    subset_interpolations = False
    if subset_interpolations:
        print('### excluding some interpolation ranges ###')
        displayed_interpolation_indices = [
            index for index in displayed_interpolation_indices
            if index < 6 or index > 8]
        print('displayed interpolations: ' + str(
            displayed_interpolation_indices))
    partial_sums: typing.List[int] = []
    partial_sums_maximum = 0.
    # Sum over all interpolations, not just the displayed ones, so the result
    #  accurately reflects the calculated attribution
    for interpolation in range(provided_interpolations_count):
        partial_sum = partial_gradients[interpolation]
        if len(partial_sums) != 0:
            partial_sum += partial_sums[-1]
        partial_sums.append(partial_sum)
        partial_sums_maximum = max(
            partial_sums_maximum, tf.math.reduce_max(tf.math.abs(partial_sum)))
    plot = image_tensors.ImagePlot()
    plot.set_default_single_channel_color_scheme(
        image_tensors.SingleDimensionSymmetricColorScheme(0., 0.95))
    # Interpolations
    for interpolation in displayed_interpolation_indices:
        if interpolated_images.shape[-1] == 1:
            plot.add_single_channel(interpolated_images[interpolation])
        else:
            plot.add_rgb_image(interpolated_images[interpolation])
    plot.new_row()
    # Certainty interpolations
    if certainty_channel:
        for interpolation in displayed_interpolation_indices:
            plot.add_single_channel(certainty_interpolations[interpolation])
        plot.new_row()
        # Overlaid
        # TODO: convert to greyscale if in color
        for interpolation in displayed_interpolation_indices:
            plot.add_overlay(
                interpolated_images[interpolation],
                tf.expand_dims(certainty_interpolations[interpolation], -1))
        plot.new_row()
    # Gradients
    max_gradient = tf.math.reduce_max(tf.math.abs(partial_gradients))
    for interpolation in displayed_interpolation_indices:
        plot.add_single_channel(
            image_tensors.brighten(
                partial_gradients[interpolation] / max_gradient, 2))
    plot.new_row()
    # Partial sums
    for interpolation in displayed_interpolation_indices:
        plot.add_single_channel(
            image_tensors.brighten(
                partial_sums[interpolation] / partial_sums_maximum, 2))
    plot.show()


# TODO: Consider replacing monte_carlo=True with interpolations_count=1
def integrated_gradients(
        images: tf.Tensor, model: keras.Model, baseline: tf.Tensor,
        monte_carlo: bool = False, baseline_samples_count: int = None,
        interpolation_count: int = 100, output_class: int = None,
        display_partial_sums: bool = False) -> tf.Tensor:

    # TODO: ADD AUTOMATIC CHECK THAT THE ATTRIBUTIONS ADD UP TO THE DIFFERENCE
    # IN SCORE AS RECOMMENDED BY https://arxiv.org/pdf/1703.01365.pdf AND IF
    # NOT RECOMMEND / AUTOMATICALLY INCREASE NUMBER OF INTERPOLATIONS

    """
        Calculate the salience of each pixel to the prediction the model makes
        for each image using the Integrated Gradients method.

        The integrated gradients method, introduced
        https://arxiv.org/pdf/1703.01365.pdf, calculates an approximated
        integral of the gradient of the model output with respect to the input
        value, from a baseline image to the subject image at each pixel,
        approximating the contribution of that pixel as a Shapley value to the
        overall prediction.

        Using multiple baselines and monte_carlo is equivalent to the Expected
        Gradients method https://arxiv.org/pdf/1906.10670.pdf.

        :param images: The images for which the salience of input pixels to the
            prediction is desired to be known, as a Tensor with shape
            ([image index], pixel row, pixel column, channel).
        :param model: The model which is making the predictions.
        :param baseline: An image to use as the "neutral" image from which to
            start the integration, as a Tensor with shape
            ([baseline index], pixel row, pixel column, channel). If the
            baseline index is present, an average of attribution over the
            baselines will be taken.
        :param monte_carlo: By default this function uses Riemann sums
            to calculate the gradient integral. If this argument is True,
            Monte Carlo integration will be used instead.
        :param baseline_samples_count: How many samples to average over.
            If this is None, random baseline sampling will be disabled and each
            baseline will be used once. If monte_carlo is enabled, a random
            interpolation will be used for each sample, the total number of
            gradient calculations will be equal to this argument (or the number
            of baselines if this argument is None). If monte_carlo is disabled,
            each baseline will have 100 equally spaced interpolations, and the
            total number of gradient calculations will be equal to 100 times
            this argument (or one hundred times the number of baselines if this
            argument is None).
        :param interpolation_count: How many interpolations are calculated
            per basline sample. This parameter is not used in monte_carlo mode.
        :param output_class: The class for which the prediction probability
            attributions are calculated. If None, the class predicted by the
            model for each image is used.
        :param display_partial_sums: If true, show on screen a plot of
            interpolated images and partial sum contributions to the
            attribution.
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """
    if monte_carlo:
        interpolation_count = 1
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
    if len(baseline.shape) > 4:
        raise ValueError('Invalid baseline shape: ' + str(baseline.shape))
    if len(baseline.shape) == 3:
        baseline = tf.expand_dims(baseline, axis=0)
    if baseline_samples_count is None:
        baseline_samples = baseline
    else:
        baseline_samples = tensor_tools.pick(baseline, baseline_samples_count)
    # Axes: (image, baseline_sample, interpolation, row, column, channel)
    image_count = images.shape[0]
    baseline_sample_count = baseline_samples.shape[0]
    row_count = images.shape[1]
    column_count = images.shape[2]
    channel_count = images.shape[3]
    evaluations_shape = [
        image_count, baseline_sample_count, interpolation_count]
    image_shape = [row_count, column_count, channel_count]
    interpolations_placeholder = tf.ones(
        [interpolation_count] + image_shape, tf.float16)
    # TODO: Allocating new axis but not tiling them to the full size
    # (broadcasting lazily) would save some unnecessary operations in later
    # steps.
    (broadcast_images, broadcast_baseline_samples, _) = \
        tensor_tools.axis_outer_operation(
            0, [images, baseline_samples, interpolations_placeholder],
            lambda tensors: tensors)
    if monte_carlo:
        interpolations = tf.reshape(
            tf.random.uniform(evaluations_shape, 0., 1.),
            evaluations_shape + [1, 1, 1])
    else:
        start = 0.5 / interpolation_count
        stop = 1. - (0.5 / interpolation_count)
        # Left un-broadcasted
        interpolations = tf.linspace(
            [[[[[start]]]]], [[[[[stop]]]]], interpolation_count, axis=2)
    deltas = broadcast_images - broadcast_baseline_samples
    interpolated_images = broadcast_baseline_samples + deltas * interpolations
    if output_class is None:
        predictions = tf.argmax(model(images), axis=-1)
    else:
        predictions = tf.fill([images.shape[0]], output_class)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        unstructured_shape = [np.prod(evaluations_shape)] + image_shape
    # TODO Code from applying softmax, assuming logits provided and
    # probabilities desired. Remove
    #    logits = model(tf.reshape(interpolated_images, unstructured_shape))
    #    unstructured_probabilities = tf.nn.softmax(logits, axis=-1)
    #    structured_probabilities = tf.reshape(
    #        unstructured_probabilities, evaluations_shape + [class_count])
        unstructured_outputs = model(tf.reshape(
            interpolated_images, unstructured_shape))
        structured_outputs = tf.reshape(
            unstructured_outputs, evaluations_shape + [class_count])
        # prediction_indices are the indices of the output values of interest
        # "4" for the three evaluations axes plus one for the coordinates
        prediction_indices = np.zeros(evaluations_shape + [4], np.int32)
        for coordinates in tensor_tools.Coordinates(evaluations_shape):
            prediction = predictions[coordinates[0]]
            prediction_indices[tuple(coordinates)] = coordinates + [prediction]
    # TODO Code from applying softmax, assuming logits provided and
    # probabilities desired. Remove
    #    prediction_probabilities = tf.gather_nd(
    #        structured_probabilities, prediction_indices)
        prediction_outputs = tf.gather_nd(
            structured_outputs, prediction_indices)
        gradients = tape.gradient(
            prediction_outputs, interpolated_images)
    # "partial_gradients" in the sense of partial derivatives.
    partial_gradients = tf.einsum('...i,...i->...', deltas, gradients)
    if display_partial_sums:
        if image_count > 1:
            raise ValueError(
                'Cannot display interpolations for multi-image attribution.')
        display_interpolations(interpolated_images, partial_gradients)
    return tf.math.reduce_mean(partial_gradients, axis=[1, 2])


def simple_integrated_gradients(
        images: tf.Tensor, model: keras.Model,
        baseline_value: typing.Union[float, tf.Tensor],
        output_class: int = None, display_partial_sums: bool = False
        ) -> tf.Tensor:
    """
        Performs Integrated Gradients where the same baseline value is used for
        each pixel.

        :param images: The images to calculate feature attribution for.
        :param model: The model to calculate feature attribution for
        :param baseline_value: The pixel value to be used as a baseline. It
            may be a single axis tensor or a number.
        :param output_class: The class probability to calculate attribution
            for. If None then the predicted (highest probability) class will
            be used.
        :param display_partial_sums: If True, images of the partial sums will
            be shown on the screen.
        :return: A tensor containing an attribution value for each pixel.
    """
    if (len(images.shape)) != 4:
        raise ValueError(
            'Image input should have shape (sample count, row count, '
            + 'column count, channel count)')
    baseline_value = tf.convert_to_tensor(baseline_value)
    if len(baseline_value.shape) > 1:
        raise ValueError('Only a single channel axis is supported.')
    if len(baseline_value.shape) == 0:
        baseline_value = tf.reshape(baseline_value, [1])
    baseline_value = tf.reshape(
        baseline_value, [1, 1, 1] + list(baseline_value.shape))
    baseline_shape = list(images.shape)
    baseline_shape[0] = 1
    # Broadcast together
    baseline = tf.zeros(baseline_shape, baseline_value.dtype) + baseline_value
    return integrated_gradients(
        images, model, baseline, output_class=output_class,
        display_partial_sums=display_partial_sums)


def double_sided_integrated_gradients(
        images: tf.Tensor, model: keras.Model, minimum: float = 0.,
        maximum: float = 1., output_class: int = None,
        display_partial_sums: bool = False) -> tf.Tensor:
    """
        Performs Integrated Gradients taking an average from the minimum
        (default 0.0) and maximum (default 1.0) baselines.
    """
    lower = simple_integrated_gradients(
        images, model, minimum, output_class=output_class,
        display_partial_sums=display_partial_sums)
    upper = simple_integrated_gradients(
        images, model, maximum, output_class=output_class,
        display_partial_sums=display_partial_sums)
    return (lower + upper) / 2.


def gaussian_integrated_gradients(
        image: tf.Tensor, model: keras.Model, standard_deviation: float = 0.1,
        minimum: float = 0., maximum: float = 1.,
        baseline_multiplicity: int = 100,
        output_class: typing.Optional[int] = None) -> tf.Tensor:
    """
        Performs integrated gradients attribution using as a baseline the
        original image with gaussian distributed noise added to each pixel.

        :param image: The image or images to calculate attribution for. A
            tensor of shape ([image index], row, column, color channel).
        :param model:
            The model to calculate attribution for.
        :param standard_deviation: How much to vary the image to create
            attribution baselines.
        :param minimum: The minimum value of the input range for the model.
            Generated baselines will be clipped to this value.
        :param maximum: The maximum value of the input range for the model.
            Generated baselines will be clipped to this value.
        :param baseline_multiplicity: How many attributions with different
            baselines to average over for each image.
        :param output_class: Which class to calculate attribution for. By
            default the predicted class is used.
        :return: A tensor showing how much each pixel support the attributed
            prediction.
    """
    baselines_shape = image.shape.as_list()
    baselines_shape[0] = baseline_multiplicity
    noise = tf.random.normal(baselines_shape, 0, standard_deviation)
    baselines = tf.maximum(minimum, tf.minimum(
        maximum, image + noise))
    return integrated_gradients(
        image, model, baselines, interpolation_count=20,
        output_class=output_class)

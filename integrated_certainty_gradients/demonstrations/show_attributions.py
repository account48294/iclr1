"""
    Display a selection of attribution methods applied to a given example for
    comparison.
"""
import typing

import keras
import tensorflow as tf

from .. import image_tensors, tensor_tools, pixel_certainty
from ..attribution import (
    feature_removal, integrated_gradients, integrated_certainty_gradients)
from ..data.dataset import Dataset


def show_attributions(
        model: keras.Model, image: tf.Tensor, dataset: Dataset,
        include_simple_feature_removal: bool = False,
        output_class: typing.Optional[int] = None) -> None:
    """
        Display a variety of feature methods.

        :param model: The model to calculate prediction feature attributions
            for.
        :param image: The input image to calculate prediction feature
            attributions for.
        :param dataset: A selection of images from the same dataset as the
            input image, for use with the Expected Gradients method.
        :param include_simple_feature_removal: If True, include simple feature
            removal attribution methods such as blacking pixels in the display.
        :param output_class: If provided, show the attribution for this class,
            which may not be the predicted one.
    """

    zero_integrated_gradient = integrated_certainty_gradients.\
        certainty_aware_simple_integrated_gradients(
            image, model, 0.0, output_class=output_class)[0]
    expected_gradients = integrated_gradients.integrated_gradients(
        image, model, dataset, True, 500, output_class=output_class)
    baseline_distribution = tensor_tools.pick(dataset, 500)
    distribution_baseline = tf.reduce_mean(baseline_distribution, 0)

    distribution_integrated_gradients = \
        integrated_gradients.integrated_gradients(
            image, model, distribution_baseline, output_class=output_class)
    # TODO: MAKE IT THE GRADIENTS FOR THE CORRECT CLASS
    gradients = integrated_gradients.classifier_gradients(image, model)
    certainty_gradients = integrated_certainty_gradients \
        .image_integrated_certainty_gradients(
            image, model, output_class=output_class)
    gaussian_attribution = integrated_certainty_gradients.\
        gaussian_certainty_gradients(
            image, model, standard_deviation=0.1, output_class=output_class)

    color_scheme = image_tensors.GreyDegenerateTriangularColorScheme(
        extremal_lightness=0.95)
    plots = (
        image_tensors.ImagePlot()
        .set_default_single_channel_color_scheme(
            color_scheme.maximum_first_channel_color_scheme())
        .set_default_two_channel_color_scheme(color_scheme)
        .add_unsigned_single_channel_scale('Value')
        .add_two_channel_scale(
            'Certainty', secondary_channel_values=[1., 0.5])
        .add_signed_single_channel_scale('Gradient')
        .add_two_channel_scale(
            'Combined', minimum=-1.0, continuous_channel=1,
            secondary_channel_values=[1., 0., -1.],
            color_scheme=image_tensors.SignedTwoDimensionalColorScheme())
    )
    greyscale_image = image[0]
    if greyscale_image.shape[-1] == 4:
        greyscale_image = pixel_certainty.collapse_value_channels(
            greyscale_image)
    plots.add_two_channel_positive_saturated(
        greyscale_image / 2. + .5,
        title='Source image')
    image_value = pixel_certainty.discard_certainty(image[0])
    show_image_value = False
    if show_image_value:
        if image.shape[-1] == 4:
            plots.add_rgb_image(
                image_value,
                title='Image value')
        else:
            plots.add_single_channel(
                pixel_certainty.discard_certainty(image[0]),
                title='Image value')
    (
        plots
        .add_single_channel(
            pixel_certainty.discard_certainty(distribution_baseline),
            title='Distribution baseline')
        .new_row()
        # .add_single_channel(
        #    discard_certainty(gradients[0]), True, title='Value gradient')
        .add_single_channel(
            pixel_certainty.discard_value(gradients[0]), True,
            title='Certainty gradient')
        # .add_two_channel_positive_white(
        #    gradients[0], True, title='Combined gradients')
        .add_single_channel(
            zero_integrated_gradient,
            True, title='Zero integrated gradients')
        .add_single_channel(
            integrated_certainty_gradients
            .certainty_aware_simple_integrated_gradients(
                image, model, 0.5, output_class=output_class)[0],
            True, title='Middle integrated gradients')
        .add_single_channel(
            integrated_certainty_gradients
            .certainty_aware_double_sided_integrated_gradients(
                image, model, output_class=output_class)[0],
            True, title='Double sided integrated gradients')
        .add_single_channel(
            expected_gradients[0], True, title='Expected gradients')
        .add_single_channel(
            distribution_integrated_gradients[0], True,
            title='Distribution integrated gradients')
        .new_row()
    )
    if include_simple_feature_removal:
        (
            plots
            .add_single_channel(
                feature_removal.simple_feature_removal(image_value, model),
                True, title='Simple feature removal')
            .add_single_channel(
                feature_removal.simple_feature_removal(
                    image_value, model, 0.5),
                True, title='Midpoint feature removal')
            .add_single_channel(
                feature_removal.double_sided_feature_removal(
                    image_value, model),
                True, title='Double sided feature removal')
            .add_single_channel(
                feature_removal.feature_certainty_removal(image[0], model),
                True, title='Feature certainty removal')
        )
    (
        plots
        .add_single_channel(
            certainty_gradients[0], True, title='Certainty gradients')
        .add_overlay(
            image_tensors.normalize_channel_centered(
                tf.expand_dims(certainty_gradients[0], -1),
                0, -1.0, 1.0, 0.),
            image_tensors.remap_channel(
                image_tensors.rgb_to_greyscale(
                    pixel_certainty.discard_certainty(image[0])),
                0, 0., 1., -0.5, 0.5),
            color_scheme=image_tensors.SignedTwoDimensionalColorScheme(),
            title='Overlaid certainty gradients A')
        .add_overlay(
            image_tensors.remap_channel(
                image_tensors.rgb_to_greyscale(
                    pixel_certainty.discard_certainty(image[0])),
                0, 0., 1., -0.5, 0.5),
            image_tensors.normalize_channel_centered(
                tf.expand_dims(certainty_gradients[0], -1),
                0, -1., 1., 0.),
            color_scheme=image_tensors.SignedTwoDimensionalColorScheme(),
            title='Overlaid certainty gradients B')
        .add_single_channel(
            gaussian_attribution[0], normalize=True,
            title='Additive Gaussian noise')
        .show()
    )

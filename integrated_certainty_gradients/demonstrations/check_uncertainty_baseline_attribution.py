"""
    Demonstration that attribution is suppressed in areas of low certainty.
"""

import tensorflow as tf
from tensorflow import keras

from .. import tensor_tools, image_tensors, pixel_certainty
from ..attribution import integrated_gradients, integrated_certainty_gradients


def check_uncertainty_baseline_attribution(
        image: tf.Tensor, model: keras.Model, band_top: int, band_width: int
        ) -> None:
    """
        Display images that show attribution is suppressed if certainty is
        reduced.

        :param image: The input image to calculate attribution for.
        :param model: The model to calculate attribution for.
        :param band_top: The distance from the top of the image at which to
            start suppressing certainty.
        :param band_width: The distance below band_top over which to suppress
            certainty.
    """
    band_bottom = band_top + band_width
    if band_bottom > image.shape[-3]:
        raise ValueError('Certainty ablation band is outside tensor volume.')
    image = (
        tensor_tools
        .Selection()[..., band_top:band_bottom, :, -1:]
        .multiplex(tf.zeros_like(image, tf.float32), image)
    )
    baseline_zero = tensor_tools.Selection()[..., 0:-1].multiplex(
        tf.zeros_like(image, tf.float32), image)
    attribution_zero = integrated_gradients.integrated_gradients(
        image, model, baseline_zero)
    baseline_one = tensor_tools.Selection()[..., 0:-1].multiplex(
        tf.ones_like(image, tf.float32), image)
    attribution_one = integrated_gradients.integrated_gradients(
        image, model, baseline_one)
    attribution_combined = attribution_one + attribution_zero
    certainty_gradients = integrated_certainty_gradients \
        .image_integrated_certainty_gradients(image, model)

    color_scheme = image_tensors.GreyDegenerateTriangularColorScheme(
        extremal_lightness=0.9)
    (
        image_tensors.ImagePlot()
        .set_default_single_channel_color_scheme(
            color_scheme.maximum_first_channel_color_scheme())
        .set_default_two_channel_color_scheme(color_scheme)
        .add_unsigned_single_channel_scale('Value')
        .add_two_channel_scale('Certainty', secondary_channel_values=[1., 0.5])
        .add_two_channel_positive_saturated(
            tf.squeeze(
                image_tensors.normalize_channel_centered(
                    pixel_certainty.collapse_value_channels(image),
                    0, 0., 1., 0.)
            ),
            title='Image')
        .add_two_channel_positive_saturated(
            tf.squeeze(
                image_tensors.normalize_channel_centered(
                    pixel_certainty.collapse_value_channels(baseline_zero),
                    0, 0., 1., 0.)
            ),
            title='Baseline Zero')
        .add_two_channel_positive_saturated(
            tf.squeeze(
                image_tensors.normalize_channel_centered(
                    pixel_certainty.collapse_value_channels(baseline_one),
                    0, 0., 1., 0.)
            ),
            title='Baseline One')
        .new_row()
        .add_signed_single_channel_scale('Attribution')
        .add_single_channel(
            tf.squeeze(certainty_gradients), True,
            title='Certainty Gradient Attribution')
        .add_single_channel(
            tf.squeeze(attribution_combined), True,
            title='Attribution Combined')
        .add_single_channel(
            tf.squeeze(attribution_zero), True, title='Attribution Zero')
        .add_single_channel(
            tf.squeeze(attribution_one), True, title='Attribution One')
        .show()
    )

"""
    Display attribution using the official implementation for the Expected
    Gradients method https://export.arxiv.org/pdf/1906.10670 published at
    https://github.com/suinleelab/path_explain to validate results are accurate
    and not due to implementation errors.
"""

import math
import typing

import tensorflow as tf
import path_explain

from .. import image_tensors


def display_expected_gradients(
        base_model: tf.keras.Model, input_tensor: tf.Tensor,
        dataset: typing.Iterable[tf.Tensor], target_class: int) -> None:
    """
        Dispaly attribution using the official implementation for the Expected
        Gradients method https://export.arxiv.org/pdf/1906.10670 published at
        https://github.com/suinleelab/path_explain

        :param base_model: The classifier model to calculate attribution for.
        :param input_tensor: The input to calculate attribution for.
        :param dataset: List of inputs to use as a baseline.
        :param target_class: The output class to calculate attribution for.
            Usually the true class for the classification task.
    """
    # Remove the batch and certainty axes
    original_input_shape = list(base_model.input_shape)[1:-1]
    flat_input_shape = [math.prod(original_input_shape)]
    dataset = [tf.reshape(tensor, flat_input_shape) for tensor in dataset]
    flattened_model = tf.keras.models.Sequential([
        # Unflatten the input
        tf.keras.layers.Reshape(
            original_input_shape, input_shape=flat_input_shape),
        # Add a certainty channel
        tf.keras.layers.Lambda(
            lambda input: tf.stack([input, tf.ones_like(input)], axis=-1),
            list(original_input_shape) + [2]),
        base_model
    ])
    explainer = path_explain.PathExplainerTF(flattened_model)
    attributions = explainer.attributions(
        inputs=tf.reshape(input_tensor, [1] + flat_input_shape),
        baseline=tf.convert_to_tensor(dataset), batch_size=1,
        num_samples=2000, use_expectation=True, output_indices=target_class)
    attributions = tf.reshape(attributions, original_input_shape + [1])
    (
        image_tensors
        .ImagePlot()
        .add_single_channel(input_tensor)
        .add_single_channel(attributions, normalize=True)
        .show()
    )

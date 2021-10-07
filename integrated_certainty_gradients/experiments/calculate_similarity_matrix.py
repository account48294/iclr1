import keras
import tensorflow as tf

from integrated_certainty_gradients import pixel_certainty, \
    tensor_normalization
from integrated_certainty_gradients.attribution import \
    integrated_certainty_gradients, integrated_gradients


def calculate_similarity_matrix(
        model: keras.Model, images: tf.Tensor, sample_size: int):
    method_names = [
        'AGN', 'middle baseline', 'ICG', 'mean baseline', 'EG']
    scores = {}
    for method in method_names:
        similarities = {}
        for other_method in method_names:
            similarities[other_method] = 0
        scores[method] = similarities
    distribution_mean = tf.reduce_mean(images, 0)
    for sample_index in range(sample_size):
        image = images[sample_index]
        image = tf.expand_dims(image, 0)
        attributions = {}
        attributions['AGN'] = integrated_certainty_gradients. \
            gaussian_certainty_gradients(
                image, model, standard_deviation=0.1)[0]
        attributions['middle baseline'] = integrated_certainty_gradients. \
            certainty_aware_simple_integrated_gradients(
                image, model, 0.5)[0]
        attributions['ICG'] = integrated_certainty_gradients \
            .image_integrated_certainty_gradients(image, model)[0]
        attributions['mean baseline'] =  integrated_gradients. \
            integrated_gradients(
                image, model, distribution_mean)[0]
        attributions['EG'] = integrated_gradients.integrated_gradients(
            image, model, images, True, 500)[0]

        for method_name in method_names:
            attributions[method_name] = tensor_normalization.normalize(
                attributions[method_name], 0.,
                tensor_normalization.euclidean_norm)

        for first_method_index in range(len(method_names)):
            for second_method_index in range(
                    first_method_index, len(method_names)):
                first_method = method_names[first_method_index]
                second_method = method_names[second_method_index]
                similarity = tf.reduce_sum(
                    attributions[first_method] * attributions[second_method])
                scores[first_method][second_method] += similarity
                if first_method != second_method:
                    scores[second_method][first_method] += similarity

        for method in method_names:
            attributions[method] = tensor_normalization.normalize(attributions[method], 0, tensor_normalization.maximum_deviation)
        from integrated_certainty_gradients import image_tensors
        plain_image = pixel_certainty.discard_certainty(image[0])
        (
            image_tensors.ImagePlot()
            .set_default_two_channel_color_scheme(image_tensors.SignedTwoDimensionalColorScheme(0.4, 0.6))
            .add_overlay(tf.expand_dims(attributions['AGN'], -1), plain_image, title='AGN')
            .add_overlay(tf.expand_dims(attributions['middle baseline'], -1), plain_image, title='middle baseline')
            .add_overlay(tf.expand_dims(attributions['ICG'], -1), plain_image, title='ICG')
            .add_overlay(tf.expand_dims(attributions['mean baseline'], -1), plain_image, title='mean baseline')
            .add_overlay(tf.expand_dims(attributions['EG'], -1), plain_image, title='EG')
            .new_row()
            .add_single_channel(attributions['AGN'])
            .add_single_channel(attributions['middle baseline'])
            .add_single_channel(attributions['ICG'])
            .add_single_channel(attributions['mean baseline'])
            .add_single_channel(attributions['EG'])
            .show()
        )

    for first_method in method_names:
        for second_method in method_names:
            scores[first_method][second_method] = \
                scores[first_method][second_method] / sample_size

    matrix = [list(row.values()) for row in scores.values()]

    print(matrix)

    matrix = tf.convert_to_tensor(matrix)

    from integrated_certainty_gradients import image_tensors

    (
        image_tensors.ImagePlot()
        .add_single_channel(matrix)
        .show()
    )

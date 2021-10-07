import typing
from collections import namedtuple

import tensorflow as tf
import keras

from integrated_certainty_gradients import (
    file_utilities, image_tensors, pixel_certainty)
from integrated_certainty_gradients.attribution import (
    integrated_certainty_gradients, integrated_gradients)
from integrated_certainty_gradients.collection_utilities import constant_dict


def set_value(
        tensor: tf.Tensor, location: typing.Tuple, new_value: float):
    array = tensor.numpy()
    array[location] = new_value
    return tf.convert_to_tensor(array)


def attribution_names():
    return ['Middle baseline', 'AGN', 'ICG', 'ECG', 'Mean baseline', 'EG',]


def calculate_attributions(
        model: keras.Model, images: typing.Dict[str, tf.Tensor],
        dataset: tf.Tensor, class_index: int) -> typing.Dict[str, tf.Tensor]:
    attributions = {}
    attributions['Middle baseline'] = (
        integrated_certainty_gradients.
        certainty_aware_simple_integrated_gradients(
            images['Middle baseline'], model, 0.5, output_class=class_index
        )[0]
    )
    attributions['AGN'] = (
        integrated_certainty_gradients.gaussian_certainty_gradients(
            images['AGN'], model, standard_deviation=0.1,
            output_class=class_index
        )[0]
    )
    attributions['ICG'] = (
        integrated_certainty_gradients.image_integrated_certainty_gradients(
            images['ICG'], model, output_class=class_index
        )[0]
    )
    attributions['ECG'] = (
        integrated_certainty_gradients.random_path_certainty_gradients(
            images['ECG'], model, 100, output_class=class_index,
            display_partial_sums=True
        )[0]
    )
    # attributions['ECG'] = (
    #    integrated_certainty_gradients.image_expected_certainty_gradients(
    #        image[0], model, 500, output_class=target_class,
    #        display_partial_sums=True
    #    )[0]
    # )
    attributions['Mean baseline'] = integrated_gradients.integrated_gradients(
        images['Mean baseline'], model, tf.reduce_mean(dataset, axis=0),
        output_class=class_index
    )[0]
    attributions['EG'] = integrated_gradients.integrated_gradients(
        images['EG'], model, dataset, True, 500, output_class=class_index
    )[0]
    return attributions


PixelAttribution = namedtuple('PixelAttribution', ['row', 'column', 'value'])


def sort_attributions(
        attributions: typing.Dict[str, tf.Tensor], least_first: bool = False
        ) -> typing.Dict[str, typing.List[PixelAttribution]]:

    # TODO: ALLOW REVERSE ORDER

    result = {}
    for name, attribution in attributions.items():
        pixel_attributions = []
        for row in range(attribution.shape[0]):
            for column in range(attribution.shape[1]):
                pixel_attributions.append(
                    PixelAttribution(row, column, attribution[row, column]))
        if least_first:
            pixel_attributions.sort(key=lambda attribution: attribution.value)
        else:
            pixel_attributions.sort(key=lambda attribution: -attribution.value)
        result[name] = pixel_attributions
    return result


def calculate_pixel_flipping(
        model: keras.Model, image: tf.Tensor, dataset: tf.Tensor,
        recalculate: bool = False, target_second_class: bool = False):
    results_directory = 'experiment_results/pixel_flipping/'
    file_utilities.ensure_directory_exists(results_directory)
    results_file_path = file_utilities.unique_path(
        results_directory + '/results', 'csv')
    with open(results_file_path, 'x') as results_file:
        results_file.write(','.join(attribution_names()))
        initial_predictions = model(image)[0]
        predicted_class_index = tf.argmax(initial_predictions).numpy()
        print('Class probabilities: ')
        print(initial_predictions)
        if target_second_class:
            second_class = tf.math.top_k(
                initial_predictions, 2).indices.numpy()[1]
            print('Second class: ' + str(second_class))
            target_class_index = second_class
        else:
            target_class_index = predicted_class_index
        target_class_vector = tf.one_hot(
            target_class_index, initial_predictions.shape[-1])

        def loss_function(predictions):
            loss = tf.keras.losses.CategoricalCrossentropy()
            return loss(target_class_vector, predictions).numpy()

        degraded_images = constant_dict(attribution_names(), image)
        pixel_attributions = None
        for iteration in range(0, 60):

            if pixel_attributions is None or recalculate:
                attributions = calculate_attributions(
                    model, degraded_images, dataset, target_class_index)
                # Using a sort to find final value is a bit slow, but it's ok
                pixel_attributions = sort_attributions(
                    attributions, least_first=target_second_class)

            results_file.write('\n')
            results = []
            plot = image_tensors.ImagePlot()
            for method_name, locations in pixel_attributions.items():
                if recalculate:
                    score = locations[0]
                else:
                    score = locations[iteration]
                # Add sample and channel axes
                location = (0,) + (score.row, score.column) + (0,)
                current_image = degraded_images[method_name]
                plot.add_single_channel(
                    pixel_certainty.discard_certainty(current_image)[0],
                    title=method_name)
                loss = loss_function(model(current_image)[0])
                print((method_name + ': ').ljust(20) + str(loss))
                results.append(str(loss))
                # Include batch and channel axes
                previous_value = current_image[location]
                if previous_value < 0.5:
                    new_value = 1.
                else:
                    new_value = 0.
                current_image = set_value(current_image, location, new_value)
                degraded_images[method_name] = current_image
            results_file.write(','.join(results))
            plot.new_row()
            for method_name in attribution_names():
                plot.add_single_channel(
                    attributions[method_name], normalize=True)
            plot.show()



# def calculate_pixel_flipping_recalculate(
#         model: keras.Model, image: tf.Tensor, dataset: tf.Tensor,
#         target_second_class: bool = False):
#     results_directory = 'experiment_results/pixel_flipping/'
#     file_utilities.ensure_directory_exists(results_directory)
#     results_file_path = file_utilities.unique_path(
#         results_directory + '/results', 'csv')
#     with open(results_file_path, 'x') as results_file:
#         results_file.write(','.join(attribution_names()))
#         initial_predictions = model(image)[0]
#         predicted_class_index = tf.argmax(initial_predictions).numpy()
#         predicted_class_vector = tf.one_hot(
#             predicted_class_index, initial_predictions.shape[-1])
#         print(initial_predictions)
#         if target_second_class:
#             second_class = tf.math.top_k(initial_predictions, 2).indices.numpy()[1]
#             second_class_vector = tf.one_hot(
#                 second_class, initial_predictions.shape[-1])
#             print('Second class: ' + str(second_class))
#             target_class_index = second_class
#             target_class_vector = second_class_vector
#         else:
#             target_class_index = predicted_class_index
#             target_class_vector = predicted_class_vector
#
#         def loss_function(predictions):
#             loss = tf.keras.losses.CategoricalCrossentropy()
#             return loss(target_class_vector, predictions).numpy()
#
#         degraded_images = {}
#         for attribution_method_name in attribution_names():
#             degraded_images[attribution_method_name] = image
#         for iteration in range(0, 60):
#             results_file.write('\n')
#             results = []
#             plot = image_tensors.ImagePlot()
#             attributions = calculate_attributions(
#                 model, degraded_images, dataset, target_class_index)
#             # Using a sort to find final value is a bit slow, but it's ok
#             bottom_locations = {
#                 method_name: (pixel.row, pixel.column)
#                 for pixel in sort_attributions(attributions)}
#             for method_name, location in bottom_locations.items():
#                 # Add sample and channel axes
#                 location = (0,) + tuple(location) + (0,)
#                 current_image = degraded_images[method_name]
#                 plot.add_single_channel(
#                     pixel_certainty.discard_certainty(current_image)[0],
#                     title=method_name)
#                 loss = loss_function(model(current_image)[0])
#                 print((method_name + ': ').ljust(20) + str(loss))
#                 results.append(str(loss))
#                 # Include batch and channel axes
#                 previous_value = current_image[location]
#                 if previous_value < 0.5:
#                     new_value = 1.
#                 else:
#                     new_value = 0.
#                 current_image = set_value(current_image, location, new_value)
#                 degraded_images[method_name] = current_image
#             results_file.write(','.join(results))
#             plot.new_row()
#             for method_name in attribution_names():
#                 plot.add_single_channel(
#                     attributions[method_name], normalize=True)
#             plot.show()
#
#
# def calculate_pixel_flipping(
#         model: keras.Model, image: tf.Tensor, dataset: tf.Tensor,
#         target_second_class: bool = False):
#
#
#
#     results_directory = 'experiment_results/pixel_flipping/'
#     file_utilities.ensure_directory_exists(results_directory)
#     results_file_path = file_utilities.unique_path(
#         results_directory + '/results', 'csv')
#     with open(results_file_path, 'x') as results_file:
#         initial_predictions = model(image)[0]
#         predicted_class_index = tf.argmax(initial_predictions).numpy()
#         predicted_class_vector = tf.one_hot(
#             predicted_class_index, initial_predictions.shape[-1])
#         print(initial_predictions)
#         if target_second_class:
#             second_class = tf.math.top_k(initial_predictions, 2).indices.numpy()[1]
#             second_class_vector = tf.one_hot(
#                 second_class, initial_predictions.shape[-1])
#             print('Second class: ' + str(second_class))
#             target_class_index = second_class
#             target_class_vector = second_class_vector
#         else:
#             target_class_index = predicted_class_index
#             target_class_vector = predicted_class_vector
#         attributions = calculate_attributions(
#             model, constant_dict(attribution_names(), image), dataset,
#             target_class_index)
#         scores = sort_attributions(attributions)
#
#         def loss_function(predictions):
#             loss = tf.keras.losses.CategoricalCrossentropy()
#             return loss(target_class_vector, predictions).numpy()
#
#         degraded_images = {}
#         for attribution_method_name in attributions:
#             degraded_images[attribution_method_name] = image
#         for iteration in range(0, 60):
#             results_file.write('\n')
#             results = []
#             plot = image_tensors.ImagePlot()
#             for method_name, attribution_scores in scores.items():
#                 current_image = degraded_images[method_name]
#                 plot.add_single_channel(
#                     pixel_certainty.discard_certainty(current_image)[0],
#                     title=method_name)
#                 loss = loss_function(model(current_image)[0])
#                 print((method_name + ': ').ljust(20) + str(loss))
#                 results.append(str(loss))
#                 score = attribution_scores[iteration]
#                 # Include batch and channel axes
#                 location = (0, score.row, score.column, 0,)
#                 previous_value = current_image[location]
#                 if previous_value < 0.5:
#                     new_value = 1.
#                 else:
#                     new_value = 0.
#                 current_image = set_value(current_image, location, new_value)
#                 degraded_images[method_name] = current_image
#             results_file.write(','.join(results))
#             plot.new_row()
#             for method_name in attribution_names():
#                 plot.add_single_channel(
#                     attributions[method_name], normalize=True)
#             plot.show()

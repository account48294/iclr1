"""
    Provides data for the Dark Squares scenario. This scenario challenges a
    model to identify whether dark (0.25 to 0.5 lightness) squares are present
    on a black (0 lightness) background, with other dark shapes and light
    squares present (the other dark shapes and light squares do not affect the
    required prediction).

    This scenario is designed to be challenging for the Expected Gradients
    attribution method because the light squares will appear as dark squares
    in the interpolations used by that method.
"""

import logging
import random
import os
import typing

import imageio
import numpy as np
import tensorflow as tf

from .. import pixel_certainty, file_utilities
from . import dataset


def load_black_white(path: str) -> np.ndarray:
    """
        Load a black and white image from a file as a Numpy array.
    """
    result = imageio.imread(path)
    result = np.array(result)
    if result.ndim == 3:
        # Convert to greyscale
        result = result[..., 0]
    result = result / 255.
    return result


def save_image(path: str, image: tf.Tensor) -> None:
    """
        Write an image tensor with values in the range 0. to 1. to a file.
    """
    imageio.imwrite(path, (image * 255).astype(np.uint8))


def load_shapes(path: str) -> typing.Dict[str, typing.List[np.ndarray]]:
    """
        Load all the "shapes" (images) from files in a directory.

        :param path: Path to a directory where the images are stored. The
            directory should contain subdirectories named according to the type
            of image they contain, and the subdirectories should directly
            contain the image files.
        :return: A map from names of the image groups to list of the images
            as Numpy arrays.
    """
    result: typing.Dict[str, typing.List[np.ndarray]] = {}

    for directory in os.scandir(path):
        if not directory.is_dir():
            continue
        result[directory.name] = []
        for file in os.scandir(directory):
            if not file.is_file():
                continue
            result[directory.name].append(load_black_white(file.path))
    return result


def place_shape(
        background: np.ndarray, shape: np.ndarray, retry: bool = False
        ) -> bool:
    """
        Attempt to place a "shape"/image into an empty (black / zero value)
        area of an existing image. The location is chosen randomly and if
        it is empty, the background image is updated to add the shape. If
        the location has any nonzero pixels the process is aborted.

        :param background: The image to attempt to change by adding the shape.
        :param shape: The image to add to the background.
        :param retry: If placing the image failed, keep retrying until it is
            placed. If there is nowhere suitable to place the image and retry
            is set to True, the call will hang.
        :return: Whether or not placing the image succeeded.
    """
    if bool(random.getrandbits(1)):
        shape = np.transpose(shape)
    if bool(random.getrandbits(1)):
        shape = np.flipud(shape)
    if bool(random.getrandbits(1)):
        shape = np.fliplr(shape)

    def try_place():
        y_position = random.randrange(0, background.shape[0] - shape.shape[0])
        x_position = random.randrange(0, background.shape[1] - shape.shape[1])
        for y_index in range(shape.shape[0]):
            for x_index in range(shape.shape[1]):
                if background[
                        y_position + y_index, x_position + x_index] != 0.:
                    return None, None
        return x_position, y_position

    while True:
        x_position, y_position = try_place()
        if x_position is not None:
            break
        if not retry:
            return False

    for y_index in range(shape.shape[0]):
        for x_index in range(shape.shape[1]):
            background[y_position + y_index, x_position + x_index] = shape[
                y_index, x_index]
    return True


def make_image(
        general_shapes: typing.List[np.ndarray],
        special_shapes: typing.List[np.ndarray],
        density: int) -> np.ndarray:
    """
        Generate an image by placing "shapes" (provided small images) on a
        black (zero value) background. The image is attempted to be uniformly
        and randomly filled to an approximate density by repeatedly placing
        the shapes in random locations. If the location is empty the shape
        is placed there, otherwise the attempt is aborted. Black (zero value)
        locations of previous shapes are considered empty. The brightness
        of each shape is randomly scaled by 0.5 to 1.0.

        :param general_shapes: A list of different shapes for which the regular
            space-filling process is used to populate the image. Any specific
            shape in the list might not be placed.
        :param special_shapes: These shapes are guaranteed to be placed in
            the image, 1 to 3 copies of each shape in the list will be placed.
            If they happen to be placed in such a way that there is no room to
            continue the process, the call will hang, so do not provide
            more shapes than can fit in the image.
        :param density: Desired density of shapes in the final image. The
            expectation for filled space increases monotonically with
            increasing density but it is not a linear relationship.
        :return: A Numpy array containing a black (zero) background overlaid
            with non-overlapping shapes/images.
    """
    image = np.zeros([50, 50])
    for special_shapes_class in special_shapes:
        for _ in range(random.randint(1, 3)):
            place_shape(
                image,
                random.choice(special_shapes_class) * random.uniform(0.5, 1.),
                True)
    for _ in range(density):
        place_shape(
            image, random.choice(general_shapes) * random.uniform(0.5, 1.))
    return image


shapes = load_shapes('datasets/shapes/shapes')
circles = [shape * 0.5 for shape in shapes['circles']]
triangles = [shape * 0.5 for shape in shapes['triangles']]
grey_quadrilaterals = [shape * 0.5 for shape in shapes['rectangles']]
white_quadrilaterals = shapes['rectangles']
null_shape = [tf.zeros([1, 1, 1])]


def generate_class_0_sample() -> tf.Tensor:
    """
         Create a 50x50 greyscale image not containing dark squares.
    """
    regular_shapes = triangles + white_quadrilaterals
    return make_image(regular_shapes, [], 10)


def generate_class_1_sample() -> tf.Tensor:
    """
         Create a 50x50 greyscale image containing dark squares.
    """
    regular_shapes = triangles + white_quadrilaterals
    return make_image(regular_shapes, [grey_quadrilaterals], 10)


def build_test_dataset() -> None:
    """
         Save a test dataset to disk for reuse.

         10,000 images of each class are saved. Files are saved under project
         directory datasets/shapes/test
    """
    file_utilities.ensure_directory_exists('datasets/shapes/test/0')
    for index in range(1, 10001):
        path = 'datasets/shapes/test/0/' + str(index) + '.png'
        logging.info('Wrote class 0 examples to %s', path)
        save_image(path, generate_class_0_sample())
    file_utilities.ensure_directory_exists('datasets/shapes/test/1')
    for index in range(1, 10001):
        path = 'datasets/shapes/test/1/' + str(index) + '.png'
        logging.info('Wrote class 1 examples to %s', path)
        save_image(path, generate_class_1_sample())


def dark_squares_dataset() -> dataset.Dataset:
    """
        50x50 pixel greyscale images consisting of squares and triangles on
        a black background, divided into two classes: images containing a dark
        square and images without a dark square.
    """

    def generator_dataset(size):

        def generate_sample():
            for _ in range(size):
                if bool(random.getrandbits(1)):
                    yield (
                        tf.expand_dims(
                            tf.convert_to_tensor(
                                generate_class_0_sample()),
                            -1),
                        tf.convert_to_tensor([1., 0.]))
                else:
                    yield (
                        tf.expand_dims(
                            tf.convert_to_tensor(
                                generate_class_1_sample()),
                            -1),
                        tf.convert_to_tensor([0., 1.]))

        return tf.data.Dataset.from_generator(
            generate_sample,
            output_signature=(
                tf.TensorSpec([50, 50, 1], tf.float32),
                tf.TensorSpec([2], tf.float32)
            ))

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory='datasets/shapes/test',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=100,
        image_size=(50, 50),
        seed=123)
    # It is a pure function so does not need special determinism accommodation.
    test_dataset = test_dataset.map(
        lambda x, y: (x / 255., y), deterministic=False)
    test_dataset = test_dataset.unbatch()
    return dataset.Dataset(generator_dataset(10000), test_dataset)


def three_dark_squares_image(
        include_top_square: bool = False, include_middle_square: bool = False,
        include_bottom_square: bool = False, with_certainty: bool = False
        ) -> tf.Tensor:
    """
        An image containing 3 dark squares, from the test dataset with index
        3 (when loaded with tf.keras.preprocessing.image_dataset_from_directory
        seed=123)

        :param with_certainty: Add a fully certain certainty channel
        :return:
    """
    image = tf.keras.preprocessing.image.load_img(
        'datasets/shapes/three_dark_squares.png', color_mode='grayscale')
    image = np.asarray(image).reshape([1, 50, 50, 1]) / 255.
    if not include_top_square:
        for y in range(0, 14):
            for x in range(0, 18):
                image[0, y, x, 0] = 0.
    if not include_middle_square:
        for y in range(15, 22):
            for x in range(0, 10):
                image[0, y, x, 0] = 0
    if not include_bottom_square:
        for y in range(30, 50):
            for x in range(0, 20):
                image[0, y, x, 0] = 0
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if with_certainty:
        image = pixel_certainty.add_certainty_channel(image)
    return image

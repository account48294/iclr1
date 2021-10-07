import typing

import tensorflow as tf

from integrated_certainty_gradients.image_tensors import image_data_to_tensor
from integrated_certainty_gradients.tests.utilities import TensorflowTestCase


class TestImageTensors(TensorflowTestCase):

    @staticmethod
    def build_image_tensor(
            count: typing.Optional[int], channels: typing.Optional[int]):
        shape = [5, 5]
        if count is not None:
            shape.insert(0, count)
        if channels is not None:
            shape.append(channels)
        return tf.fill(shape, 0.5)

    def test_four_axis_multiple(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(3, 3), False),
            self.build_image_tensor(3, 3))

    def test_batch_axis_multiple(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(None, 3), False),
            self.build_image_tensor(1, 3))

    def test_channel_axis_multiple(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(3, None), False),
            self.build_image_tensor(3, 1))

    def test_two_axis_multiple(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(None, None), False),
            self.build_image_tensor(1, 1))

    def test_four_axis_single(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(1, 3)),
            self.build_image_tensor(None, 3))

    def test_batch_axis_single(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(None, 3)),
            self.build_image_tensor(None, 3))

    def test_channel_axis_single(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(1, None)),
            self.build_image_tensor(None, 1))

    def test_two_axis_single(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(None, None)),
            self.build_image_tensor(None, 1))

    def test_numpy(self):
        self.assert_tensor_equal(
            image_data_to_tensor(self.build_image_tensor(None, None).numpy()),
            self.build_image_tensor(None, 1))

    def test_lists(self):
        self.assert_tensor_equal(
            image_data_to_tensor([[0.5] * 5] * 5),
            self.build_image_tensor(None, 1))

    def test_integers(self):
        self.assert_tensor_equal(
            image_data_to_tensor([[128] * 5] * 5),
            tf.fill((5, 5, 1), 128. / 255.))


import tensorflow as tf

import integrated_certainty_gradients.tensor_normalization as normalization

from .utilities import TensorflowTestCase


class TestNormalization(TensorflowTestCase):

    def test_constructor(self):
        normalization.Normalizer()

    def test_static(self):
        self.assert_tensor_equal(
            normalization.normalize(
                tf.convert_to_tensor([-5., -1., 3., 5.]),
                1, 2),
            tf.convert_to_tensor([-2., 0., 2., 3.]))

    def test_standard_deviation(self):
        self.assert_tensor_equal(
            normalization.normalize(
                tf.convert_to_tensor([2., -3., 0., 2., 3., 2.]),
                normalization.mean, normalization.root_second_moment,
                recenter=True),
            tf.convert_to_tensor([0.5, -2., -0.5, 0.5, 1., 0.5]))

    def test_absolute_value(self):
        self.assert_tensor_equal(
            normalization.normalize(
                tf.convert_to_tensor([-2., 1., -1., 0]),
                0, normalization.maximum_deviation),
            tf.convert_to_tensor([-1, .5, -.5, 0]))

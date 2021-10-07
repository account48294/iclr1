"""
    Tools for normalizing the range of Tensorflow tensors.
"""

import typing

import tensorflow as tf

from integrated_certainty_gradients import tensorflow_utils


NumberStatistic = typing.Union[
    typing.Callable[[tf.Tensor], tensorflow_utils.Number],
    tensorflow_utils.Number]


def midpoint(data: tf.Tensor) -> tf.Tensor:
    """ Half way between the minimum and maximum values in the tensor. """
    return (tf.reduce_max(data) - tf.reduce_min(data)) / 2


def mean(data: tf.Tensor) -> tf.Tensor:
    """ The arithmetic mean of the values in the tensor. """
    return tf.reduce_mean(data)


def zero(data: tf.Tensor) -> tf.Tensor:
    """ The constant function 0. """
    return tf.convert_to_tensor(0, dtype=data.dtype)


def full_range(data: tf.Tensor) -> tf.Tensor:
    """ The difference between the smallest and larget value in the tensor. """
    return tf.reduce_max(data) - tf.reduce_min(data)


def maximum_deviation(data: tf.Tensor) -> tf.Tensor:
    """ The largest unsigned distance of any value in the tensor from 0. """
    return tf.reduce_max(tf.abs(data))


def mean_absolute_deviation(data: tf.Tensor) -> tf.Tensor:
    """
        The arithmetic mean of the distances of each value in the tensor from
        0.
    """
    return tf.reduce_mean(tf.abs(data))


def root_second_moment(data: tf.Tensor) -> tf.Tensor:
    """
        The population standard deviation when the pivot is the mean.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(data)))


def euclidean_norm(data: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_sum(tf.square(data)))


def normalize(
        data: tf.Tensor,
        pivot: NumberStatistic,
        dispersion: NumberStatistic,
        recenter: bool = False,
        target_dispersion: tensorflow_utils.Number = 1.,
        minimum_cutoff: typing.Optional[tensorflow_utils.Number] = None,
        maximum_cutoff: typing.Optional[tensorflow_utils.Number] = None,
        cutoff_throw_exception: bool = True
        ) -> tf.Tensor:
    """
        Apply the action of the Normalizer class in a single function call.
    """
    return (
        Normalizer()
        .set_pivot(pivot, recenter)
        .set_dispersion(dispersion, target_dispersion)
        .set_cutoffs(minimum_cutoff, maximum_cutoff, cutoff_throw_exception)
    )(data)


class Normalizer:
    """
        Fits data into a different range by recentering and scaling.

        Use example include moving data from the range 0,1 to the range -1,1
        and shrinking unbounded data into the range 0,1.
    """

    def __init__(self):
        self._pivot_callback = None
        self._recenter = False
        self._dispersion_callback = None
        self._target_dispersion = 1.
        self._maximum_cutoff = None
        self._minimum_cutoff = None
        self._on_cutoff_throw_exception = True

    def set_pivot(
            self, pivot: tensorflow_utils.Number,
            recenter: bool = False) -> 'Normalizer':
        """
             Provide the value which is fixed when doing normalisation. This
             can be either a callback that operates on the Tensor, or a simple
             number.
        """
        if callable(pivot):
            pivot_callback = pivot
        else:
            def pivot_callback(data: tf.Tensor):
                return pivot
        self._pivot_callback = pivot_callback
        self._recenter = recenter
        return self

    def set_dispersion(
            self, dispersion: tensorflow_utils.Number,
            target_dispersion: tensorflow_utils.Number = 1.) -> 'Normalizer':
        """
             Provide the measure of dispersion to scale by when doing
             normalisation. This can be either a callback that operates on the
             Tensor, or a simple number.
        """
        if callable(dispersion):
            dispersion_callback = dispersion
        else:
            def dispersion_callback(data: tf.Tensor):
                return dispersion
        self._dispersion_callback = dispersion_callback
        self._target_dispersion = target_dispersion
        return self

    def __call__(self, data: tf.Tensor) -> tf.Tensor:
        """
             Apply the configured normalization operation to a tensor.
        """
        if self._pivot_callback is None:
            raise Exception('Set the pivot before using the normalizer.')
        if self._dispersion_callback is None:
            raise Exception('Set the dispersion before using the normalizer.')
        pivot = self._pivot_callback(data)
        data = data - pivot
        dispersion = self._dispersion_callback(data)
        scaled = data * self._target_dispersion / dispersion
        if not self._recenter:
            scaled = scaled + pivot
        if self._on_cutoff_throw_exception:
            if (
                    self._minimum_cutoff is not None
                    and tf.reduce_min(scaled) < self._minimum_cutoff):
                raise ValueError('Scaled data outside range')
            if (
                    self._maximum_cutoff is not None
                    and tf.reduce_max(scaled) > self._maximum_cutoff):
                raise ValueError('Scaled data outside range')
        else:
            if self._minimum_cutoff is not None:
                scaled = tf.maximum(self._minimum_cutoff, scaled)
            if self._maximum_cutoff is not None:
                scaled = tf.minimum(self._maximum_cutoff, scaled)
        return scaled

    def set_cutoffs(
            self, minimum: tensorflow_utils.Number,
            maximum: tensorflow_utils.Number,
            throw_exception: typing.Optional[bool] = None) -> 'Normalizer':
        """
            Prevent data falling outside a range, either by throwing an
            exception if it does, or clipping any values that lie outside.
        """
        self._minimum_cutoff = minimum
        self._maximum_cutoff = maximum
        if throw_exception is not None:
            self._on_cutoff_throw_exception = throw_exception
        return self

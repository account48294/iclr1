"""
    This module provides capabilities for working with image tensors. These
    are represented as tensors whose last three axes are row, column, and
    channel respectively. The channel axis commonly represents red, green, blue
    components when of size 3 and lightness/brightness when of size 1. Tensors
    with two channels are also supported, and shown in false color.

    Color terminology:
        Hue: The 'type' of color, for example red, green, or orange.
        Lightness: From black to white, with fully chromatic colors equal to
            grey.
        Chroma: How close to a bright, pure color it is (distance from black,
            white, and grey).
        Intensity: How different the color is from grey. Black, white, and
            fully chromatic colors are all high intensity.
        Saturation: How different the color is from white (purity of color).
            Sometimes the term is used differently. It will not be used in the
            public API of this module.
        Value: Sometimes used as equivalent to lightness. Sometimes used to
            mean how different the color is from black, fully chromatic colors
            equal to white. It will not be used in this module.
        References:
            https://munsell.com/color-blog/difference-chroma-saturation/
            https://en.wikipedia.org/wiki/HSL_and_HSV
"""


import typing
import math

import imageio
from matplotlib import pyplot
import numpy as np
import tensorflow as tf

from integrated_certainty_gradients import tensor_tools, tensorflow_utils


# Various types which can represent raster image data.
ImageData = typing.Union[
    tf.Tensor,
    np.ndarray,
    typing.List[typing.List[tensorflow_utils.Number]],
    typing.List[typing.List[typing.List[tensorflow_utils.Number]]],
    typing.List[typing.List[typing.List[typing.List[tensorflow_utils.Number]]]]
]


# TODO: ADD OPTION TO REPLACE OUT-OF-RANGE PIXELS WITH A COLOR (OR MAYBE
#  1 FOR TOO HIGH, 1 FOR TOO LOW).
def image_data_to_tensor(
        image_data: ImageData,
        single: bool = True,
        channel_count: typing.Union[None, int, typing.List[int]] = [1, 3],
        validate_range: bool = True,
        throw_exception: bool = True,
        image_name: str = None
        ) -> typing.Optional[tf.Tensor]:
    """
        Convert a variety of representations of image data to a standardised
        Tensor form.

        :param image_data: Raster image data, represented by either a
            tensorflow Tensor, numpy ndarray, or native Python nested lists.
            The data can have a single greyscale channel or 3 color channels:
            red, green, blue. The data in any type must have 2 to 4 'axes':
            "sample" (optional), "row", "column", "color channel" (optional).
            If present, the sample axis must have one dimension, and will be
            removed. The color channel axis must have one or thee dimensions,
            and if not present, a single dimension axis will be created.
        :param single: Is the data for just one image. If true a 3 axis tensor
            will be returned and an exception will be thrown if there is more
            than single image data. If false a 4 axis tensor will be returned
            whether the data is for a single image or multiple.
        :param channel_count: The numbers of channels which are allowed. For
            example [1, 3] could indicate greyscale or red-green-blue images.
            If None then any number of channels are allowed. This will be used
            to guess whether the tensor has a batch or channel axis if it has
            three axes.
        :param validate_range: If True the values of the tensor will be checked
            to confirm they are between 0 and 1. If the values are integers
            they will be assumed to represent byte values and be divided by
            255.
        :param throw_exception: If True an exception will be thrown if the
            data is not in the correct format. If False, None will be returned
            instead.
        :param image_name: Used for reporting errors if provided.
        :return: A Tensor with axes [batch], row, column, color channel. The
            batch axis will only be present if the "single" parameter is False.
    """
    # single is True by default because exploratory / interactive programming
    #  is probably more likely to use single images, whereas batch images
    #  are more likely to be in programs. Therefore, it is more helpful
    #  to reduce the code size for single image calls.

    def descriptor() -> str:
        result = 'Image data'
        if image_name is not None:
            result += ' "' + image_name + '"'
        return result

    def invalid(message):
        if throw_exception:
            raise ValueError(descriptor() + ' ' + message)

    if isinstance(channel_count, int):
        channel_count = [channel_count]

    result = tf.convert_to_tensor(image_data)
    shape = tf.shape(result)
    if len(shape) == 4:
        if single:
            result = tf.squeeze(result, [0])
    elif len(shape) == 3:
        if channel_count is not None:
            batch_axis_likely = shape[2] not in channel_count
        else:
            batch_axis_likely = shape[0] == 1
        if batch_axis_likely:
            if single:
                result = tf.squeeze(result, [0])
            result = tf.expand_dims(result, -1)
        elif not single:
            result = tf.expand_dims(result, 0)
    elif len(shape) == 2:
        if not single:
            result = tf.expand_dims(result, 0)
        result = tf.expand_dims(result, -1)
    else:
        return invalid('did not have expected shape for image.')
    if channel_count is not None and tf.shape(result)[-1] not in channel_count:
        return invalid('did not have the specified number of color channels.')
    if not (result.dtype.is_floating or result.dtype.is_integer):
        return invalid('does not have an appropriate dtype.')
    if validate_range:
        if result.dtype.is_integer:
            result = (
                tf.cast(result, tensorflow_utils.default_float_type()) / 255.)
        maximum = tf.reduce_max(result)
        if maximum > 1.:
            return invalid(
                'value out of range (pixel too bright): '
                + str(maximum.numpy()))
        minimum = tf.reduce_min(result)
        if minimum < 0.:
            return invalid(
                ' value out of range (pixel too dark): '
                + str(minimum.numpy()))
    return result


def hcl_to_rgb(colors: ImageData) -> tf.Tensor:
    """
        Convert a tensor of colors from Hue Chroma Lightness representation
        to Red Green Blue representation.

        Hue Chroma Lightness is ambiguous: it is not possible to have maximum
        chroma and maximum or minimum lightness simultaneously. This HCL space
        treats lightness as primary: lightness will be accurate and chroma
        will be scaled according to the maximum achievable with a given
        lightness.

        :param colors: A tensor of colors with the last axis having three
            channels: hue, chroma and  lightness.
        :return: A tensor of colours with the last axis having three
            channels: red, green and  blue.
    """
    colors = image_data_to_tensor(colors)

    # Algorithm from
    # https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
    def hue_to_rgb(p, q, t):
        t = tf.where(t < 0, t + 1, t)
        t = tf.where(t > 1, t - 1, t)
        lower = tf.where(t < 1/6, p + (q - p) * 6 * t, q)
        upper = tf.where(t < 2/3, p + (q - p) * (2 / 3 - t) * 6, p)
        return tf.where(t < 0.5, lower, upper)

    hue, saturation, lightness = tf.unstack(colors, axis=-1)
    q = tf.where(
        lightness < 0.5,
        lightness * (1 + saturation),
        lightness + saturation - lightness * saturation)
    p = 2 * lightness - q
    achromatic = tf.stack([lightness, lightness, lightness], axis=-1)
    red = hue_to_rgb(p, q, hue + 1 / 3)
    green = hue_to_rgb(p, q, hue)
    blue = hue_to_rgb(p, q, hue - 1 / 3)
    chromatic = tf.stack([red, green, blue], axis=-1)
    result = tf.where(
        tf.expand_dims(saturation, -1) == 0, achromatic, chromatic)
    # Small numeric inaccuracies might push the value out of range.
    return tf.maximum(tf.minimum(result, 1.), 0.)


def rgb_to_hcl(colors: ImageData) -> tf.Tensor:
    """
        Convert a tensor of colors from Red Green Blue representation to Hue
        Chroma Lightness.

        Hue Chroma Lightness is ambiguous: it is not possible to have maximum
        chroma and maximum or minimum lightness simultaneously. This HCL space
        treats lightness as primary: lightness will be accurate and chroma
        will be scaled according to the maximum achievable with a given
        lightness.

        :param pixels: A tensor of colors with the last axis having three
            channels: red, green and  blue.
        :return: A tensor of colours with the last axis having three
            channels: hue, chroma and  lightness.
    """
    colors = image_data_to_tensor(colors)

    # Algorithm from
    # https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion

    maximum = tf.math.reduce_max(colors, axis=-1)
    minimum = tf.math.reduce_min(colors, axis=-1)
    red, green, blue = tf.unstack(colors, axis=-1)
    lightness = (maximum + minimum) / 2
    difference = maximum - minimum
    saturation = tf.where(
        maximum == minimum,
        0.,
        tf.where(
            lightness > 0.5,
            difference / (2. - maximum - minimum),
            difference / (maximum + minimum)
        )
    )
    # Default case, blue is maximum
    hue = (red - green) / difference + 4.
    # Where green is maximum
    hue = tf.where(green == maximum, (blue - red) / difference + 2., hue)
    # Where red is maximum
    hue = tf.where(
        red == maximum,
        (green - blue) / difference + tf.cast(
            tf.where(green < blue, 6., 0.), colors.dtype),
        hue)
    # Where it is achromatic
    hue = tf.where(maximum == minimum, 0., hue)
    hue = hue / 6.
    return tf.stack([hue, saturation, lightness], axis=-1)


class ColorScheme:
    """
        Convert tensors with a given number of channels to color (three
        channel) tensors. These can be channel counts which do not have a
        direct interpretation os colors (for example two channels), or when the
        channels do have a direct color interpretation, mapping the color to
        one that makes the data easier to interpret (false color).

        Color schemes attempt to produce a result which is as easy for
        colorblind people to perceive as possible for any individual condition.
    """

    def map(self, values: typing.Any) -> tf.Tensor:
        """ Apply the color scheme to an input. """
        if not isinstance(values, tf.Tensor):
            return self.map(tf.convert_to_tensor(values)).numpy()
        minimum = self.minimum()
        while len(minimum.shape) < len(values.shape):
            minimum = tf.expand_dims(minimum, 0)
        maximum = self.maximum()
        while len(maximum.shape) < len(values.shape):
            maximum = tf.expand_dims(maximum, 0)
        if tf.reduce_any(tf.cast(values, tf.float32) < minimum):
            raise ValueError(
                'Tensor values out of range for color scheme: below minimum '
                + str(self.minimum().numpy()))
        if tf.reduce_any(tf.cast(values, tf.float32) > maximum):
            raise ValueError(
                'Tensor values out of range for color scheme: above maximum '
                + str(self.maximum().numpy()))
        if self._tensorized():
            if len(values.shape) == 0:
                values = tf.reshape(values, [1])
            return self._map(values)
        else:
            dtype = values.dtype
            if len(values.shape) == 0:
                pixels_shape = []
                value_list = tf.reshape(values, [1])
            else:
                pixels_shape = values.shape.as_list()[:-1]
                value_list = tf.reshape(
                #   Replace math.prod with np.prod for older python
                #   compatibility values,
                #   [math.prod(pixels_shape), values.shape[-1]])
                    values, [np.prod(pixels_shape), values.shape[-1]])

            def tensorflow_map():
                array = value_list.numpy()
                mapping = np.vectorize(
                    lambda value: np.array(self._map(value.tolist())),
                    signature='(m)->(n)')
                return tf.convert_to_tensor(mapping(array), dtype=dtype)

            return tf.reshape(
                tf.py_function(tensorflow_map, inp=[], Tout=dtype),
                pixels_shape + [3])

    def __call__(self, value):
        return self.map(value)

    def minimum(self):
        """ The minimum input channel values for this color scheme. """
        return tf.convert_to_tensor(self._minimum())

    def maximum(self):
        """ The maximum input channel values for this color scheme. """
        return tf.convert_to_tensor(self._maximum())

    def _map(self, values):
        raise NotImplementedError

    def _minimum(self):
        raise NotImplementedError

    def _maximum(self):
        raise NotImplementedError

    def _tensorized(self):
        return False


class SingleDimensionColorScheme(ColorScheme):
    """ Abstract class for color schemes that apply to single channel data. """

    def _map(self, values):
        if self._tensorized():
            raise NotImplementedError
        return self._map_value(values[0])

    def _map_value(self, value):
        raise NotImplementedError


class TwoDimensionColorScheme(ColorScheme):
    """ Abstract class for color schemes that apply to two channel data. """

    def first_dimension(
            self, intercept: float = None) -> SingleDimensionColorScheme:
        """
            Return a color scheme across the first dimension at the given
            intercept of the second dimension.
        """
        # TODO: implement
        raise NotImplementedError

    def second_dimension(
            self, intercept: float = None) -> SingleDimensionColorScheme:
        """
            Return a color scheme acroos the second dimension at the given
            intercept of the first dimension.
        """
        # TODO: implement
        raise NotImplementedError


class GreyscaleColorScheme(SingleDimensionColorScheme):
    """ Convert single channel data to three channel greyscale. """

    def _map(self, values):
        shape = values.shape.as_list()
        shape[-1] = 3
        return values + tf.zeros(shape, values.dtype)

    def _minimum(self):
        return 0.

    def _maximum(self):
        return 1.

    def _tensorized(self):
        return True


class SingleDimensionSymmetricColorScheme(SingleDimensionColorScheme):
    """
        Maps values in the range -1.0 to 1.0. Negative values have a different
        hue to positive values. Chroma and lightness vary continuously between
        a neutral value (at 0) and an extremal value (at 1 and -1). The hue
        is opposite in the negative and positive regions.
    """
    def __init__(
            self, neutral_lightness: float = 0.,
            extremal_lightness: float = 1.,
            constant_chroma: typing.Optional[bool] = None,
            positive_hue_angle: float = 30.):
        """

            :param neutral_lightness: The lightness at value 0.
            :param extremal_lightness: The lightness at value 1 and -1.
            :param constant_chroma: If True the chroma will be 1,
                otherwise it will be zero at value zero and 1 at value -1 and
                1.
            :param positive_hue_angle: The hue angle in degrees identifying
                the color hue of positive values.
        """
        self._neutral_lightness = neutral_lightness
        self._extremal_lightness = extremal_lightness
        if constant_chroma is None:
            # This clause avoids a discontinuity in the color mapping by
            # default. Either the neutral color has lightness 0 or 1 (black
            # or white), or else the chroma can vary and the neutral
            # color has chroma 0 (a grey).
            if neutral_lightness in (0., 1.):
                self._constant_chroma = True
            else:
                self._constant_chroma = False
        else:
            self._constant_chroma = constant_chroma
        self._positive_hue = positive_hue_angle / 360.

    def _map(self, values):
        # Design constraints: Brightness (sum of color values) varies linearly
        # with input value. Chroma varies continuously with input value. At
        # extremal values chroma is the maximum possible for the given
        # lightness.
        values = tf.squeeze(values, -1)
        hue = tf.where(
            values > 0.0, self._positive_hue, self._positive_hue + 0.5)
        hue = tf.cast(hue, values.dtype)
        absolute_values = tf.abs(values)
        lightness = (
            absolute_values * self._extremal_lightness
            + (1. - absolute_values) * self._neutral_lightness)
        if self._constant_chroma:
            chroma = tf.ones_like(values)
        else:
            chroma = absolute_values
        return hcl_to_rgb(tf.stack([hue, chroma, lightness], axis=-1))

    def _minimum(self):
        return -1.

    def _maximum(self):
        return 1.

    def _tensorized(self):
        return True


class SignedTwoDimensionalColorScheme(TwoDimensionColorScheme):

    """
        Maps two channel values with ranges -1. to 1. to colors, with (0., 0.)
        mapped to grey.

        Chroma increases with the absolute value of the first channel, with
        negative values blue and positive values orange. Lightness increases
        with the value of the second channel, from minimum lightness at -1.
        value to maximum lightness at 1.
    """

    def __init__(
            self, minimum_lightness: float = 0.2,
            maximum_lightness: float = 0.8,
            variation: int = 1) -> None:
        """

            :param minimum_lightness: The lightness when the second channel has
                value -1.
            :param maximum_lightness: The lightness when the second channel has
                value 1.
            :param variation: Deprecated. which variation of hues to use.
                Must be 1, 2, or 3. 1 gives the standard orange and blue hues.
        """
        self._minimum_lightness = minimum_lightness
        self._maximum_lightness = maximum_lightness
        if variation not in [1, 2, 3]:
            raise ValueError('variation must be an integer 1, 2, or 3.')
        self._variation = variation

    def _map(self, values):
        # Change the range of the first channel to [0.0, 1.0]
        first = values[0] * 0.5 + 0.5
        # Compressing the range of the second channel makes it easier to see
        # the first channel by preventing brightness saturating.
        second = (
            self._minimum_lightness
            + (self._maximum_lightness - self._minimum_lightness)
            * values[1])
        if second > 0:
            components = [
                first + (np.int8(1) - first) * second,
                np.float32(0.5) * (np.int8(1) + second),
                (np.int8(1) - first) + first * second
            ]
        else:
            second = second + np.float32(1.)
            components = [
                first * second,
                np.float32(0.5) * second,
                (np.int8(1) - first) * second
            ]
        return np.array(change_pixel_color_scheme(
            components, self._variation))

    def _minimum(self):
        return [-1., -1.]

    def _maximum(self):
        return [1., 1.]


class TriangularColorScheme(TwoDimensionColorScheme):
    """
        A color scheme mapping the first channel (index 0) to chroma and
        the second (index 1) to lightness. Color difference are proportional
        to lightness (second channel value).

        At the minimum of the second channel color differences are at their
        least, and this color scheme is intended to represent a degenerate
        point at the minimum of the second channel. Extremal values of the
        first channel are always fully saturated when the second channel is
        maximal.
    """

    def __init__(
            self, degenerate_lightness: float = 0.2,
            neutral_lightness: float = 0.5, positive_hue_angle: float = 30.
            ) -> None:
        """

            :param degenerate_lightness: The lightness when the second channel
                has minimum value. More extremal (closer to 0. or 1.) values
                will give clearer differentiation of the second channel, but
                will make it harder to distinguish first channel values when
                the second channel is near the minimum.
            :param neutral_lightness: The lightness when the second channel
                has maximum value and the first channel has median value.
                Values closer to 0.5 will give more constant lightness for
                any value of the second channel, but values further from 0.5
                (in the opposite direction to the degenerate lightness) will
                more clearly distinguish the second channel.
            :param positive_hue_angle: The hue angle to determine positive and
                negative hues. The default gives orange/blue.
        """
        self._top_color_scheme = SingleDimensionSymmetricColorScheme(
            neutral_lightness=neutral_lightness, extremal_lightness=0.5,
            positive_hue_angle=positive_hue_angle)
        self._degenerate_lightness = degenerate_lightness

    def maximum_first_channel_color_scheme(self):
        """
            Get the single channel color scheme that is equivalent to the
            second (index 1) channel when the first channel (index 0) is set to
            its maximum value.
        """
        return self._top_color_scheme

    def _map(self, values):
        values_0, values_1 = tf.unstack(values, axis=-1)
        values_0 = tf.expand_dims(values_0, -1)
        red_green_blue = self._top_color_scheme(values_0 * 2. - 1.)
        hue, chroma, lightness = tf.unstack(
            rgb_to_hcl(red_green_blue), axis=-1)
        lightness = values_1 * lightness \
            + (1. - values_1) * self._degenerate_lightness
        return hcl_to_rgb(tf.stack([hue, chroma, lightness], -1))

    def _minimum(self):
        return [0., 0.]

    def _maximum(self):
        return [1., 1.]

    def _tensorized(self):
        return True


class GreyDegenerateTriangularColorScheme(TwoDimensionColorScheme):
    """
        A color scheme which has intensity depending on the second channel
        (with index 1), and is greyer when the second channel has value 0.

        Lightness varies with the first channel and the hue (typically blue
        or orange) changes when the first channel is above or below its
        median value. this color scheme is intended to represent a degenerate
        point at the minimum of the second channel. It has the advantage of
        allowing greater differentiation of the first channel when the second
        channel is maximum by allowing a greater range of lightnesses, at the
        cost of reduced differentiation in the second channel.
    """

    def __init__(
            self, neutral_light: bool = False, extremal_lightness: float = 0.5,
            minimum_chroma: float = 0.1, positive_hue_angle: float = 30.
            ) -> None:
        """

            :param neutral_light: If True the median value of the first
                channel will result in a color from grey to white. If False
                it will be from grey to black.
            :param extremal_lightness: The lightness when the second channel
                is at a maximum and the
            :param minimum_chroma:
            :param positive_hue_angle: The hue angle to determine positive and
                negative hues. The default gives orange/blue.
        """

        if neutral_light:
            neutral_lightness = 1.
        else:
            neutral_lightness = 0.
        self._top_color_scheme = SingleDimensionSymmetricColorScheme(
            neutral_lightness=neutral_lightness,
            extremal_lightness=extremal_lightness,
            positive_hue_angle=positive_hue_angle)
        self._minimum_chroma = minimum_chroma

    def maximum_first_channel_color_scheme(self):
        """
            Get the single channel color scheme that is equivalent to the
            second (index 1) channel when the first channel (index 0) is set to
            its maximum value.
        """
        return self._top_color_scheme

    def _map(self, values):
        values_0, values_1 = tf.unstack(values, axis=-1)
        values_0 = tf.expand_dims(values_0, -1)
        # TODO: WAS CHANGED TO self._top_color_scheme(values_0 * 2 - 1)
        # TODO: RESOLVE THE RANGES ISSUE
        saturated = self._top_color_scheme(values_0 * 2 - 1)
        grey = tf.convert_to_tensor(0.5, dtype=values.dtype)
        shifted_chroma = tf.expand_dims(
            self._minimum_chroma
            + (1. - self._minimum_chroma) * values_1,
            axis=-1)
        return shifted_chroma * saturated \
            + (1. - shifted_chroma) * grey

    def _minimum(self):
        return [0., 0.]

    def _maximum(self):
        return [1., 1.]

    def _tensorized(self):
        return True


class ImagePlot:
    """
        Convenience class to display images on screen using matplotlib pyplot.

        All add image methods take the following optional parameters:
            title: str: Display this text above the image.
            xaxis: bool: Display an x axis scale if true.
            yaxis: bool: Display a y axis scale if true.
            xlabel: str: Display this text next to the x axis.
            ylabel: str: Display this text next to the y axis.
    """

    def __init__(self):
        self._images = []
        self._parameters = []
        self._positions = {}
        self._next_position = [0, 0]
        self._rows = 1
        self._columns = 1
        self._default_two_channel_color_scheme = TriangularColorScheme()
        self._default_single_channel_color_scheme = \
            SingleDimensionSymmetricColorScheme()

    def set_default_single_channel_color_scheme(
            self, color_scheme: SingleDimensionColorScheme) -> 'ImagePlot':
        """
            Set the default (false color) color scheme for single channel
            (greyscale) images.
        """
        self._default_single_channel_color_scheme = color_scheme
        return self

    def set_default_two_channel_color_scheme(
            self, color_scheme: TwoDimensionColorScheme) -> 'ImagePlot':
        """
            Set the default color scheme for images with two color channels.
        """
        self._default_two_channel_color_scheme = color_scheme
        return self

    def show(self) -> None:
        """
            Display the images on the screen.
        """
        _, axes = pyplot.subplots(self._rows, self._columns)
        if self._rows == 1:
            axes = [axes]
        if self._columns == 1:
            axes = list(map(lambda axes: [axes], axes))
        for rows in axes:
            for current_axes in rows:
                # This sets default behaviour for submitted axes and also hides
                # 'missing' axes
                current_axes.set_visible(False)
        for index, image in enumerate(self._images):
            try:
                position = self._positions[index]
                current_axes = axes[position[0]][position[1]]
                current_axes.set_visible(True)
                parameters = self._parameters[index]
                if 'ymin' in parameters:
                    y_minimum = parameters['ymin']
                else:
                    y_minimum = 0.0
                if 'ymax' in parameters:
                    y_maximum = parameters['ymax']
                else:
                    y_maximum = 1.0
                if 'xmin' in parameters:
                    x_minimum = parameters['xmin']
                else:
                    x_minimum = 0.0
                if 'xmax' in parameters:
                    x_maximum = parameters['xmax']
                else:
                    shape = image.shape
                    x_maximum = (y_maximum - y_minimum) * shape[1] / shape[0]
                current_axes.imshow(
                    image, extent=[x_minimum, x_maximum, y_minimum, y_maximum])
                current_axes.get_xaxis().set_visible(False)
                current_axes.get_yaxis().set_visible(False)
                if 'xaxis' in parameters and parameters['xaxis']:
                    current_axes.get_xaxis().set_visible(True)
                if 'yaxis' in parameters and parameters['yaxis']:
                    current_axes.get_yaxis().set_visible(True)
                if 'xlabel' in parameters:
                    current_axes.get_xaxis().set_label_text(
                        parameters['xlabel'])
                if 'ylabel' in parameters:
                    current_axes.get_yaxis().set_label_text(
                        parameters['ylabel'])
                if 'title' in parameters:
                    current_axes.set_title(parameters['title'])
            except Exception as exception:
                message = 'Exception showing graph number ' + str(index + 1)
                if 'title' in parameters:
                    message += ' titled "' + str(parameters['title']) + '"'
                message += ': ' + str(exception)
                raise Exception(message) from exception
        pyplot.show()

    def add_signed_single_channel_scale(
            self, label: str = None, minimum: float = -1.0,
            maximum: float = 1.0,
            color_scheme: SingleDimensionColorScheme = None) -> 'ImagePlot':
        """

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (blue) end of
                the scale.
            :param maximum: The displayed value for the maximum (orange) end of
                the scale
            :return: self
        """
        if color_scheme is None:
            color_scheme = self._default_single_channel_color_scheme
        image = np.linspace(
            np.outer([1.0] * 20, [1.0]),
            np.outer([1.0] * 20, [-1.0]),
            256)
        self.add_single_channel(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum,
            color_scheme=color_scheme)
        return self

    def add_unsigned_single_channel_scale(
            self, label: str = None, minimum: float = 0.0,
            maximum: float = 1.0,
            color_scheme: SingleDimensionColorScheme = None) -> 'ImagePlot':
        """

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum of the scale.
            :param maximum: The displayed value for the maximum of the scale.
            :return: self
        """
        if color_scheme is None:
            color_scheme = self._default_single_channel_color_scheme
        image = np.linspace(
            np.outer([1.0] * 20, [1.0]),
            np.outer([1.0] * 20, [0.0]),
            256)
        self.add_single_channel(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum,
            color_scheme=color_scheme)
        return self

    def add_two_channel_scale(
            self, label: str = None, minimum: float = 0.0,
            maximum: float = 1.0, continuous_channel: int = 1,
            secondary_channel_values: typing.List[float] = [1., 0.],
            color_scheme: TwoDimensionColorScheme = None) -> 'ImagePlot':
        """
            Add a scale for color schemes representing two dimensional values.

            A series of adjacent bars will be displayed, varying continuously
            across values for one channel, each having varied values in another
            channel.

            :param label: A textual label that is shown with this scale.
            :param minimum: The minimum value for the continuous channel.
            :param maximum: The maximum value for the continuous channel.
            :param continuous_channel: Which dimension of the channel axis
                to vary continuously (0 or 1). The other will have discrete
                variations.
            :param secondary_channel_values: The values to use for the axis
                which is being shown with discrete values.
            :param color_scheme: The color scheme to encode the values. By
                default the two channel color scheme of the image plot is used.
            :return: The same image plot object.
        """
        bands = []
        minimum_value = [1.0, 1.0]
        minimum_value[continuous_channel] = minimum
        for secondary_channel_value in secondary_channel_values:
            maximum_value = [secondary_channel_value, secondary_channel_value]
            maximum_value[continuous_channel] = maximum
            minimum_value = [secondary_channel_value, secondary_channel_value]
            minimum_value[continuous_channel] = minimum
            bands.append(np.linspace(
                np.outer([1.0] * 20, maximum_value),
                np.outer([1.0] * 20, minimum_value),
                256))
        image = tf.concat(bands, 1)
        self.add_two_channel_positive_saturated(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum,
            color_scheme=color_scheme)
        return self

    def add_single_channel(
            self, image: ImageData, normalize: bool = False,
            color_scheme: SingleDimensionColorScheme = None,
            **parameters: typing.Any) -> 'ImagePlot':
        """
            Display a single channel image, in false color. The image can have
            positive or negative values.

            :param image: The image to display.
            :param normalize: If true, the values in the image will be scaled
                based on greatest absolute value whilst preserving 0.0.
            :param neutral_lightness: Sets the lightness for greyscale pixels
                with value 0.
            :param color_scheme: An integer 1, 2, or 3. Each color scheme is
                individually distinguishable for color blind viewers, but
                distinguishing between schemes may not be possible.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        image = image_data_to_tensor(
            image, validate_range=False, channel_count=1,
            image_name=parameters.get('title', None))
        if image.shape[-1] != 1:
            raise ValueError(
                'The image has multiple channels. Shape: '
                + str(image.shape))
        if color_scheme is None:
            color_scheme = self._default_single_channel_color_scheme
        if normalize:
            image = normalize_channel_centered(image, 0, -1.0, 1.0, 0.0)
        return self.add_rgb_image(color_scheme(image), **parameters)

    # TODO: rename this method to just add_two_channel and ensure is general
    #  enough
    def add_two_channel_positive_saturated(
            self, image: ImageData,
            color_scheme: TwoDimensionColorScheme = None,
            **parameters: typing.Any) -> 'ImagePlot':
        """
            Display a two unsigned channel image.

            Warmer colors from blue to orange correspond to increasing values
            in the first channel. Brighter colors from black to fully saturated
            correspond to increasing values in the second channel.
            :param image: The image to display.
            :param parameters: See ImagePlot class docstring.
            :param color_scheme: An integer 1, 2, or 3. Each color scheme is
                individually distinguishable for color blind viewers, but
                distinguishing between schemes may not be possible.
            :return: self
        """
        image_tensor = image_data_to_tensor(
            image, channel_count=2, validate_range=False,
            image_name=parameters.get('title', None))
        if color_scheme is None:
            color_scheme = self._default_two_channel_color_scheme
        return self.add_rgb_image(color_scheme(image_tensor), **parameters)

    # TODO: remove this method
    def add_two_channel_positive_white(
            self, image: ImageData, normalize: bool = False,
            **parameters: typing.Any) -> 'ImagePlot':
        """
            Display a two signed channel image.

            Warmer colors from blue to orange correspond to increasing values
            in the first channel. Lighter colors from black to white correspond
            to increasing values in the second channel.
            :param image: The image to display.
            :param normalize: If true, the values in the image will be scaled
                based on greatest absolute value of their channel whilst
                preserving 0.0.
            :param color_scheme: An integer 1, 2, or 3. Each color scheme is
                individually distinguishable for color blind viewers, but
                distinguishing between schemes may not be possible.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        image_tensor = image_data_to_tensor(
            image, channel_count=2, validate_range=False,
            image_name=parameters.get('title', None))
        if normalize:
            image = normalize_channel_centered(image, 0, -1.0, 1.0, 0.0)
            image = normalize_channel_centered(image, 1, -1.0, 1.0, 0.0)
        return self.add_two_channel_positive_saturated(
            image_tensor, color_scheme=SignedTwoDimensionalColorScheme(),
            **parameters)

    def add_overlay(
            self, first_image: ImageData, second_image: ImageData,
            **parameters: typing.Any) -> 'ImagePlot':
        """
            Displays two single signed channel images overlaid on each other
            using color mapping.

            :param first_image: The data to form the first channel of the
                displayed image.
            :param second_image: The data to form the second channel of the
                displayed image.
            :param parameters: Extra parameters which are described in the
                class docstring.
            :return: self
        """
        first_image = image_data_to_tensor(
            first_image, validate_range=False, channel_count=1)
        second_image = image_data_to_tensor(
            second_image, validate_range=False, channel_count=1)
        overlaid_image = tf.concat([first_image, second_image], -1)
        return self.add_two_channel_positive_saturated(
            overlaid_image, **parameters)

    def new_row(self) -> 'ImagePlot':
        """
            Place all additional images on a following row.

            :return: self
        """
        self._next_position = [self._next_position[0] + 1, 0]
        self._rows += 1
        return self

    def add_rgb_image(
            self, image: ImageData, **parameters: typing.Any) -> 'ImagePlot':
        """
            Add a standard RGB (red green blue) images

            :param image: The image to display.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        image = image_data_to_tensor(
            image, channel_count=3, validate_range=False,
            image_name=parameters.get('title', None))
        self._images.append(image)
        self._parameters.append(parameters)
        # Assign the current index to the current position
        self._positions[len(self._images) - 1] = tuple(self._next_position)
        self._next_position[1] = self._next_position[1] + 1
        if self._next_position[1] > self._columns:
            self._columns = self._next_position[1]
        return self


def save_image(path: str, image: ImageData) -> None:
    """
        Write an image tensor with values in the range 0. to 1. to a file.
    """
    tensor = image_data_to_tensor(image, True, validate_range=True)
    imageio.imwrite(
        path,
        tf.math.round((tensor * 255)).numpy().astype(np.uint8)
    )


def load_black_white(
        path: str, as_tensor: bool = True, batch_axis: bool = False
        ) -> typing.Union[np.ndarray, tf.Tensor]:
    """
        Load a black and white image from a file as a Numpy array.
    """
    result = imageio.imread(path)
    result = np.array(result)
    if result.ndim == 3:
        # Convert to greyscale
        result = result[..., 0]
    if batch_axis:
        result = np.expand_dims(result, 0)
    result = result / 255.
    if as_tensor:
        result = tf.convert_to_tensor(result)
    return result


def demonstrate_color_schemes() -> None:
    """
        Display plots which show the differences between the two channel color
        schemes.
    """

    plot = ImagePlot()
    color_schemes = {
        'Signed': SignedTwoDimensionalColorScheme(),
        'Triangular': TriangularColorScheme(),
        'Triangular\nvaried lightness': TriangularColorScheme(
            neutral_lightness=.2, degenerate_lightness=.8),
        'Grey Triangular': GreyDegenerateTriangularColorScheme(),
        'Grey Triangular\nextended lightness':
            GreyDegenerateTriangularColorScheme(
                neutral_light=True, extremal_lightness=0.2),
    }
    for title, color_scheme in color_schemes.items():
        minimum = color_scheme.minimum()
        maximum = color_scheme.maximum()
        plot.add_two_channel_positive_saturated(
            tf.linspace(
                tf.linspace(
                    (maximum[0], maximum[1]), (minimum[0], maximum[1]), 255),
                tf.linspace(
                    (maximum[0], minimum[1]), (minimum[0], minimum[1]), 255),
                255
            ),
            color_scheme=color_scheme, title=title
        )
    plot.show()


def channel_absolute(image: tf.Tensor, channel: int) -> tf.Tensor:
    """ Make the values in a channel positive. """
    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(tf.abs, image)
    )


def remap_channel(
        image: tf.Tensor, channel: int, input_minimum: float,
        input_maximum: float, output_minimum: float, output_maximum: float
        ) -> tf.Tensor:
    """
        Shift and scale a channel so it fits into a new range.

        :param image: The image or images tensor.
        :param channel: The index of the channel in the last axis to transform.
        :param input_minimum: The bottom of the range in the input tensor.
        :param input_maximum: The top of the range in the input tensor.
        :param output_minimum: The bottom of the range in the output tensor,
            mapped from input_minimum.
        :param output_maximum: The top of the range in the output tensor,
            mapped from input_maximum.
        :return: The transformed tensor.
    """
    if input_minimum >= input_maximum:
        raise ValueError(
            'The minimum of the input range must be less than the maximum.')

    def remap(data):
        output_range = output_maximum - output_minimum
        input_range = input_maximum - input_minimum
        scale_factor = output_range / input_range
        return (data - input_minimum) * scale_factor + output_minimum

    return tensor_tools.Selection().select_channel(-1, channel).transform(
        remap, image)


def normalize_channel_centered(
        image: tf.Tensor, channel: int, minimum: float, maximum: float,
        center_from: float) -> tf.Tensor:
    """
        Scale the channel values to fill the given range as much as possible,
            whilst preserving the center.

        :param image: The image or images tensor to transform.
        :param channel: The channel to transform in the last axis of the image
            tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :param center_from: The value of the center/origin of the data before
            the mapping. This will be mapped to the
            middle of the output range.
        :return: The transformed tensor.
    """

    def normalize_centered(data):
        centered_data = data - center_from
        data_range = tf.reduce_max(tf.abs(centered_data))
        if not data_range == 0:
            normalized_data = centered_data / data_range
        else:
            normalized_data = centered_data
        # Divide by 2 because normalized_data is from -1.0 to 1.0
        scaled_data = normalized_data * (maximum - minimum) / 2.0
        return scaled_data + ((maximum + minimum) / 2.0)

    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(normalize_centered, image)
    )


def normalize_channel_full_range(
        image: tf.Tensor, channel: int, minimum: float, maximum: float
        ) -> tf.Tensor:
    """
        Scale the channel values to fill the given range fully.

        :param image: The image or images tensor to transform.
        :param channel: The channel to transform in the last axis of the image
            tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :return: The transformed tensor.
    """

    def normalize_full_range(data):
        start_minimum = tf.reduce_min(data)
        start_maximum = tf.reduce_max(data)
        if start_minimum == start_maximum:
            scale_factor = 1
        else:
            scale_factor = (
                    (maximum - minimum) / (start_maximum - start_minimum))
        return (data - start_minimum) * scale_factor + minimum

    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(normalize_full_range, image)
    )


def normalize_channels_full_range(
        image: tf.Tensor, channels: typing.Iterable[int], minimum: float,
        maximum: float) -> tf.Tensor:
    """
        Scale the channel values to fill the given range fully. This
        implementation may not be efficient.

        :param image: The image or images tensor to transform.
        :param channels: The channels to transform in the last axis of the
            image tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :return: The transformed tensor.
    """
    start_minimum = tf.reduce_min(tf.gather(image, channels, axis=-1))
    start_maximum = tf.reduce_max(tf.gather(image, channels, axis=-1))
    if start_minimum == start_maximum:
        scale_factor = 1
    else:
        scale_factor = (maximum - minimum) / (start_maximum - start_minimum)

    def normalize_full_range(data):
        return (data - start_minimum) * scale_factor + minimum

    for channel in channels:
        selection = tensor_tools.Selection().select_channel(-1, channel)
        image = selection.transform(normalize_full_range, image)
    return image


# TODO: Deprecated, provide positive_hue_angle argument instead.
def change_pixel_color_scheme(pixel, color_scheme):
    """
        Converts a color from the basic color-blind color scheme (blue to
        orange) to a secondary color-blind color scheme.

        Not that although each color scheme is designed to be individually
        distinguishable for color blind viewers, the schemes may not be
        distinguishable between each other.

        :param pixel: A pixel in the blue-orange color scheme, represented
            as a list of floats, [red, green, blue].
        :param color_scheme: The color scheme to use: 1: blue-orange,
            2: purple-green
            3: fuchsia-green
        :return:
    """
    if color_scheme == 1:
        return pixel
    # Green is more intense that red and blue: reduce it if it is one of the
    # main colors.
    reduce_green = 0.8 + 0.2 * (pixel[0] + pixel[1] + pixel[2]) / 3
    if color_scheme == 2:
        return [pixel[1], pixel[0] * reduce_green, pixel[2]]
    if color_scheme == 3:
        return [pixel[2], pixel[0] * reduce_green, pixel[1]]
    raise ValueError('Invalid color scheme number.')


def rgb_to_greyscale(image: tf.Tensor) -> tf.Tensor:
    """
        Convert a three channel red-green-blue image tensor to a single channel
        greyscale image tensor.
    """
    return tf.reduce_mean(image, axis=-1, keepdims=True)


def brighten(image: tf.Tensor, extent: int = 2) -> tf.Tensor:
    """
        Shifts the values of a tensor away from 0.0 and towards -1.0 and 1.0.
        This may help make faint images clearer.

        :param image: The image to increase brightness of.
        :param extent: How much to increase the brightness.
        :return: The brightened image.
    """
    result = tf.math.sin(image * math.pi / 2)
    if extent == 1:
        return result
    else:
        return brighten(result, extent - 1)

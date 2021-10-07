import numpy as np
from integrated_certainty_gradients import image_tensors


def show_interpolated_squares(interpolation_position: float) -> None:
    true_image = np.zeros([100, 100])
    for row in range(10,50):
        for column in range(10,50):
            true_image[row][column] = 1.
    for row in range(80,90):
        for column in range(80,90):
            true_image[row][column] = 1.
    baseline_image = np.random.rand(100,100)
    interpolation = (
        interpolation_position * true_image
        + (1 - interpolation_position) * baseline_image)
    (
        image_tensors.ImagePlot()
        .set_default_single_channel_color_scheme(
            image_tensors.GreyscaleColorScheme())
        .add_single_channel(interpolation)
        .show()
    )

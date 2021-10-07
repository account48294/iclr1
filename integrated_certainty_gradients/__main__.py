"""
    Demonstration of the use of the Integrated Certainty Gradients feature
    attribution method and related methods and analysis.
"""


if __package__ is None:
    # Executed as a script, so let's set up the path
    import os
    import sys
    from pathlib import Path
    PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_DIRECTORY))
    os.chdir(str(PROJECT_DIRECTORY))


import tensorflow as tf


from integrated_certainty_gradients.data import (
    artificial_uncertainty, mnist, dark_squares)
from integrated_certainty_gradients.demonstrations import (
    show_attributions, check_uncertainty_baseline_attribution)
from integrated_certainty_gradients import (
    # fixpatches,
    model_tools, pixel_certainty, train_model)

# fixpatches.apply_fixpatches()

TRAIN_MODEL = False
DISPLAY_IMAGES = True

MODEL = model_tools.load_latest_model('models/active/model')

plain_dataset = artificial_uncertainty.with_certainty(
    dark_squares.dark_squares_dataset())
augmented_dataset = artificial_uncertainty.mixed_damage(
    dark_squares.dark_squares_dataset())


if TRAIN_MODEL:
    train_model.train_model(MODEL, plain_dataset, augmented_dataset)

if DISPLAY_IMAGES:
    SAMPLE_INDEX = 4
    SHOW_BASELINE_TEST = False
    INCLUDE_SIMPLE_FEATURE_REMOVAL = False

    sample = plain_dataset.sample_test_at(
        SAMPLE_INDEX, add_sample_channel=True)

    image = sample[0]  # Take x (input) value
    dataset = plain_dataset.take_test_x(2500)

    MNIST_IN_SPACE_TEST = False
    if MNIST_IN_SPACE_TEST:
        image = artificial_uncertainty.with_certainty(
            mnist.mnist_in_space(True)).sample_test_at(
                SAMPLE_INDEX, add_sample_channel=True)[0]

    print('true class:')
    print(sample[1][0])
    print('considering confidence:')
    results = MODEL.predict(image)
    print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
    print(results)
    print('ignoring confidence:')
    image_value = pixel_certainty.disregard_certainty(image)
    results = MODEL.predict(image_value)
    print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
    print(results)
    print('baseline values:')
    print(MODEL.predict(pixel_certainty.disregard_certainty(image, 0.)))

    if SHOW_BASELINE_TEST:
        check_uncertainty_baseline_attribution.\
            check_uncertainty_baseline_attribution(image, MODEL, 12, 8)

    show_attributions.show_attributions(
        MODEL, image, dataset, INCLUDE_SIMPLE_FEATURE_REMOVAL)

print('Finished')

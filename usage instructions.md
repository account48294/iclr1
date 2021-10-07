
# Usage instructions

## Setup

Tested with Python version 3.8.0.

Before using this code, install the required modules listed under `install_requires` in the `setup.py` file.
This can be done with the command `pip install -r requirements.txt` from the project directory.
To avoid dependency conflicts a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) is recommended.

Run the code by executing the `__main__.py` script in the `integrated_certainty_gradients` directory: `python3 __main__.py`.
Equivalently, execute the integrated_certainty_gradients module `python3 -m integrated_certainty_gradients`.

## Usage

By default the codebase is set up to work on a model in the
`models/active` directory. The model should be in Saved Model format
(https://www.tensorflow.org/guide/saved_model) and named "model".

Run the code by executing the `__main__.py` script in the `integrated_certainty_gradients` directory:
`python3 __main__.py`

## Changing scenario

In addition to moving a correct model to the `models/active` directory, the
dataset specified in the `__main__.py` file must be changed. These are the lines:

```
plain_dataset = data.artificial_uncertainty.with_certainty(
    dark_squares.dark_squares_dataset())
augmented_dataset = data.artificial_uncertainty.mixed_damage(
    dark_squares.dark_squares_dataset())
```

Import the appropriate packages then change
`dark_squares.dark_squares_dataset()` to the appropriate dataset for the
scenario.

## Usage tips

Several variables in `__main__.py` can be used to change the run configuration.

- Use the variables `TRAIN_MODEL` and `DISPLAY_IMAGES` to enable training a new
  or existing model or to analyse an existing model with the attribution
  methods. It is usually desirable to have only one of these options enabled.

- In DISPLAY_IMAGES mode setting the variable `INCLUDE_SIMPLE_FEATURE_REMOVAL`
  to True will include simple feature ablation attribution methods in the
  output. Setting `SHOW_BASELINE_TEST` to True will display the zero certainty
  baseline test before the attribution outputs.

- The `SAMPLE_INDEX` assignment in `__main__.py` can be varied to choose to view
  various images from the test dataset.

The local minimum scenario does not have a predefined test dataset.
You can create test samples by hand, changing the `sample` assignment in
`__main__.py`.
Function `example` in package `data.local_minimum` can help with this,
for example:

```
import data.local_minimum as local_minimum
sample = local_minimum.example(
    [0.1, 0.2, 0.3, 0.6, 0.7, 0.9], True, True, True)
```

## Local minimum scenario

To attribute the vector based Local Minimum model, it must be adapted to take
images, because the attribution tools are designed for images.

The module `vector_to_image` contains functions to assist this. Use
`vector_to_image.vector_to_image_classifier` to convert the vector model into
an image based model. For example:

```
import vector_to_image
MODEL = vector_to_image.vector_to_image_classifier(
    model_tools.load_latest_model('models/local minimum/model'))
```

The data must also be converted to images. Use the
`vector_to_image.vector_to_image_dataset` function. Apply it before adding
certainty or damage. For example:

```
import vector_to_image
    plain_dataset = data.artificial_uncertainty.with_certainty(
        vector_to_image.vector_to_image_dataset(
            data.local_minimum.local_minimum_dataset()))
```
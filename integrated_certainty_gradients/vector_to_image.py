"""
    Utilities for representing vectors (1 axis) as images (2 axis, with
    an axis of size 1) for high level APIs (datasets and models).
"""


import tensorflow as tf
from tensorflow import keras


def vector_to_image_dataset(dataset):
    """ Converts a vector dataset to 1xn size images """
    def convert_to_image(vector):
        shape = vector.shape.as_list()
        shape.insert(0, 1)
        return tf.reshape(vector, shape)
    return dataset.map_x(convert_to_image)


def vector_to_image_classifier(model: keras.Model) -> keras.Model:
    """ Converts a vector classifier to a 1xn size image classifier """
    model_input = keras.Input((1,) + model.input_shape[1:])
    # Squeeze axis 1 because axis 0 will be the batch axis.
    reshaped = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(
        model_input)
    output = model(reshaped)
    return tf.keras.Model(inputs=[model_input], outputs=[output])

"""
    Helper method with convenient configuration for training a model.
"""

from tensorflow import keras

from integrated_certainty_gradients import file_utilities, model_tools


def train_model(model, plain_dataset, augmented_dataset, batch_size=128):
    """
        Train a model with artificial uncertainty semantics.
    """
    epochs = 3
    statistics_period_batches = 100
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    plain_dataset = plain_dataset.prepare(batch_size)
    augmented_dataset = augmented_dataset.prepare(batch_size)

    file_utilities.ensure_directory_exists('statistics')
    model.fit(
        augmented_dataset.train(),
        # batch_size=batch_size,
        epochs=epochs, verbose=1,
        callbacks=[
            model_tools.BatchCountCallback(),
            # TODO: These degrade performance, possibly due to large test set
            # size. It may be due to trying to load whole epoch with generator
            # training set. Or perhaps due to .shuffle in dataset.
            # model_tools.PeriodicCallback(
            #     model_tools.DatasetEvaluator(
            #         'val', augmented_dataset.test()),
            #     statistics_period_batches),
            # model_tools.PeriodicCallback(
            #     model_tools.DatasetEvaluator(
            #         'undamaged', plain_dataset.test()),
            #     statistics_period_batches),
            model_tools.PeriodicCallback(
                model_tools.SafeCSVLogger('statistics/statistics'),
                statistics_period_batches),
            model_tools.PeriodicCallback(
                model_tools.ModelCheckpointSequential('models/active/model'),
                statistics_period_batches),
            model_tools.NonAveragingProgbarLogger(count_mode='steps'),
            # keras.callbacks.TensorBoard(
            #     log_dir='tensorboard', histogram_freq=1)
        ]
    )

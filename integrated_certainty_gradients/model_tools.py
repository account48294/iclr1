"""
    Tools for working with Keras models.
"""

import pathlib

import tensorflow as tf
from tensorflow import keras

from . import file_utilities, language


class PeriodicCallback(tf.keras.callbacks.Callback):
    """
        Call a keras callback periodically based on number of mini-batches
        trained.
    """

    def __init__(
            self, base_callback: tf.keras.callbacks.Callback,
            period_batches: int,
            period_mode: str = 'epoch') -> None:
        """

            :param base_callback: The callback to invoke periodically.
            :param period_batches: The number of batches between each call.
            :param period_mode: How to invoke the base callback. Currently only
                'epoch' mode is supported: call the base callback as if it is
                the end of an epoch.
        """
        super().__init__()
        if period_mode != 'epoch':
            raise NotImplementedError(
                'PeriodCallback only supports epoch mode.')
        self._base_callback = base_callback
        self._period_batches = period_batches
        self._current_period_batch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self._current_period_batch_count += 1
        self._base_callback.on_batch_end(batch, logs)
        if self._current_period_batch_count == self._period_batches:
            self._base_callback.on_epoch_end(self._epoch, logs)
            self._current_period_batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self._current_period_batch_count += 1
        self._base_callback.on_batch_end(batch, logs)
        if self._current_period_batch_count == self._period_batches:
            self._base_callback.on_epoch_end(self._epoch, logs)
            self._current_period_batch_count = 0


delegated_methods = [
    'set_params', 'set_model', 'on_batch_begin', 'on_train_begin',
    'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
    'on_predict_end']
for method_name in delegated_methods:
    language.delegate_to_attribute(
        PeriodicCallback, '_base_callback', method_name)
disabled_methods = [
    'on_epoch_end', 'on_train_batch_begin',
    'on_test_batch_begin', 'on_test_batch_end',
    'on_predict_batch_begin', 'on_predict_batch_end']
for method_name in disabled_methods:
    setattr(PeriodicCallback, method_name, language.do_nothing)


class NonAveragingProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    """
        A version of the keras Progbar progress tracker that shows raw, not
        averaged statistics.
    """

    def __init__(self, count_mode='samples'):
        super().__init__(count_mode)

    def on_train_batch_end(self, batch, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_test_batch_end(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_predict_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_epoch_end(epoch, logs)

    def on_test_end(self, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_test_end(logs)

    def on_predict_end(self, logs=None):
        self._update_stateful_metrics(logs)
        return super().on_predict_end(logs)

    def _update_stateful_metrics(self, logs):
        if logs is not None:
            self.progbar.stateful_metrics.update(logs.keys())


class DiagnosticCallback(keras.callbacks.Callback):
    """
        Print out (to console) general information about training progress at
        the end of each batch.
    """

    def on_train_batch_end(self, batch, logs=None):
        print('')
        print('batch end')
        print('batch: ' + str(batch))
        print('params: ' + str(self.params))
        print('logs: ' + str(logs))
        print('metrics: ' + str(self.model.metrics_names))
        print('')


class BatchCountCallback(keras.callbacks.Callback):
    """
        Count the number of batches that have occurred since training started.

        The count is not reset between epochs.
    """

    def on_train_begin(self, logs=None):
        self._batches = 0

    def on_train_batch_begin(self, batch, logs=None):
        self._batches += 1

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            logs['train_batch_count'] = self._batches

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['train_batch_count'] = self._batches


class DatasetEvaluator(keras.callbacks.Callback):
    """
        Evaluate model performance against a dataset at the end of each epoch.

        This is useful when there is more than one evaluation dataset
        (Tensorflow 3.8 Keras supports only 1 evaluation dataset). The results
        are stored as normal training logs/scalars.
    """

    def __init__(self, dataset_name: str, dataset: tf.data.Dataset) -> None:
        super().__init__()
        self._name = dataset_name
        self._dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        score = self.model.evaluate(
            self._dataset, verbose=0)
        logs[self._name + '_loss'] = score[0]
        logs[self._name + '_accuracy'] = score[1]


class SafeCSVLogger(keras.callbacks.CSVLogger):
    """
        Save model training statistics to a CSV file, ensuring previous
        statistics are not overwritten.

        If existing statistics files are found, a new sequentially numbered
        file will be created for this run.
    """

    def __init__(self, base_path: str, separator: str = ',') -> None:
        pathlib.Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        filename = file_utilities.unique_path(base_path, 'csv')
        # Creating the file ensures that another process does not think the
        # filename is available before the first epoch completes.
        open(filename, 'w').close()
        # append should be False so the header row is written.
        super().__init__(filename, separator, append=False)


class ModelCheckpointSequential(keras.callbacks.ModelCheckpoint):
    """
        Save the model being trained in its current state without overwriting
        previous saved files.

        Number saved models sequentially to ensure saved models are not
        overwritten. This is guaranteed even if training is aborted and
        restarted.
    """

    def __init__(self, base_path: str, wait_epochs: int = 0) -> None:
        super().__init__(None)
        self._base_path = base_path
        self._wait_epochs = wait_epochs
        self._waited_epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        self.filepath = file_utilities.unique_path(self._base_path, '')
        if self._waited_epochs == self._wait_epochs:
            super().on_epoch_end(epoch, logs)
            self._waited_epochs = 0
        else:
            self._waited_epochs += 1


def safe_save_model(model: keras.Model, base_path: str) -> None:
    """
        Save the model auto-creating a directory so it will not overwrite or
        be blocked by an existing model.

        :param model: The model to save.
        :param base_path: The directory in which to save the model.
    """
    keras.models.save_model(
        model,
        file_utilities.unique_path(base_path, ''),
        overwrite=False,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


def load_latest_model(base_path: str) -> keras.Model:
    """
        Load the latest model saved by safe_save_model in a directory.
    """
    return keras.models.load_model(
        file_utilities.latest_version(base_path, ''))

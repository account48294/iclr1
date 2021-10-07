"""
    This module applies fixes to required libraries which do not include the
    fixes in the published version that is used (version 3.8 in the case of
    Tensorflow).
"""

import time
import typing

import numpy as np
from tensorflow import keras

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
import tensorflow.python.distribute.coordinator.cluster_coordinator as \
     coordinator_lib
from tensorflow.python.framework import ops


def apply_fixpatches():
    """
        Monkey patches needed upstream fixes which are not yet published.
    """

    # Fixes from 29c4b893f00aad811a0366622bd93d82ed46d665
    # https://github.com/tensorflow/tensorflow/commit/29c4b893f00aad811a0366622bd93d82ed46d665

    def sync_to_numpy_or_python_type(tensors):
        """
            Syncs and converts a structure of `Tensor`s to `NumPy` arrays or
            Python scalar types. For each tensor, it calls `tensor.numpy()`.
            If the result is a scalar value, it converts it to a Python type,
            such as a float or int, by calling `result.item()`.
            Numpy scalars are converted, as Python types are often more
            convenient to deal with. This is especially useful for bfloat16
            Numpy scalars, which don't support as many operations as other
            Numpy values. Async strategies (such as `TPUStrategy` and
            `ParameterServerStrategy`) are forced to sync during this process.

            :param tensors: A structure of tensors.
            :return: `tensors`, but scalar tensors are converted to Python
                types and non-scalar tensors are converted to Numpy arrays.
        """
        if isinstance(tensors, coordinator_lib.RemoteValue):
            return tensors.fetch()

        def _to_single_numpy_or_python_type(t):
            if isinstance(t, ops.Tensor):
                x = t.numpy()
                return x.item() if np.ndim(x) == 0 else x
            return t  # Don't turn ragged or sparse tensors to NumPy.

        return nest.map_structure(_to_single_numpy_or_python_type, tensors)

    def _process_logs(self, logs):
        """Turns tensors into numpy arrays or Python scalars if necessary."""
        if logs is None:
            return {}
        return tf_utils.sync_to_numpy_or_python_type(logs)

    def _call_batch_hook_helper(self, hook_name, batch, logs):
        """Helper function for `on_*_batch_*` methods."""
        if self._check_timing:
            start_time = time.time()

        logs = self._process_logs(logs)
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)

        if self._check_timing:
            if hook_name not in self._hook_times:
                self._hook_times[hook_name] = []
            self._hook_times[hook_name].append(time.time() - start_time)

    def on_epoch_end(self: typing.Any, epoch: int, logs: dict = None) -> None:
        """
            Calls the `on_epoch_end` methods of its callbacks.
            This function should only be called during TRAIN mode.

            :param epoch: Integer, index of epoch.
            :param logs: Dict, metric results for this training epoch, and for
                the validation epoch if validation is performed. Validation
                result keys are prefixed with `val_`.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    tf_utils.sync_to_numpy_or_python_type = sync_to_numpy_or_python_type
    callback_list_class = keras.callbacks.CallbackList
    callback_list_class._process_logs = _process_logs
    callback_list_class._call_batch_hook_helper = _call_batch_hook_helper
    callback_list_class.on_epoch_end = on_epoch_end

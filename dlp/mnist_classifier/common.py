from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from keras.datasets import mnist  # Use `keras` for Keras >= 3, or use `tensorflow.keras` for Keras < 3

F32Array: TypeAlias = NDArray[np.float32]
U8Array: TypeAlias = NDArray[np.uint8]


def load_data() -> tuple[tuple[F32Array, U8Array], tuple[F32Array, U8Array]]:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    assert train_images.shape == (60000, 28, 28) and train_images.dtype == np.uint8
    assert train_labels.shape == (60000,) and train_labels.dtype == np.uint8
    assert test_images.shape == (10000, 28, 28) and test_images.dtype == np.uint8
    assert test_labels.shape == (10000,) and test_labels.dtype == np.uint8

    train_images = train_images.reshape((train_images.shape[0], -1)).astype(np.float32) / 255.0
    test_images = test_images.reshape((test_images.shape[0], -1)).astype(np.float32) / 255.0
    return (train_images, train_labels), (test_images, test_labels)

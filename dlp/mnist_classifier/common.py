from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Protocol

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from keras.datasets import mnist  # Use `keras` for Keras >= 3, or use `tensorflow.keras` for Keras < 3

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
N_FEATURES = IMAGE_HEIGHT * IMAGE_WIDTH
N_CLASSES = 10
HIDDEN_LAYER_SIZE = 512
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001  # 1e-3

type F32Array = NDArray[np.float32]
type U8Array = NDArray[np.uint8]

_raw_data: tuple[tuple[U8Array, U8Array], tuple[U8Array, U8Array]] | None = None


def load_raw_data() -> tuple[tuple[U8Array, U8Array], tuple[U8Array, U8Array]]:
    global _raw_data

    if _raw_data is not None:
        return _raw_data

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    assert train_images.shape == (60000, IMAGE_HEIGHT, IMAGE_WIDTH) and train_images.dtype == np.uint8
    assert train_labels.shape == (60000,) and train_labels.dtype == np.uint8
    assert test_images.shape == (10000, IMAGE_HEIGHT, IMAGE_WIDTH) and test_images.dtype == np.uint8
    assert test_labels.shape == (10000,) and test_labels.dtype == np.uint8

    _raw_data = (train_images, train_labels), (test_images, test_labels)
    return _raw_data


def preprocess_data(images: U8Array) -> F32Array:
    return images.reshape((images.shape[0], -1)).astype(np.float32) / np.float32(255.0)


def load_data() -> tuple[tuple[F32Array, U8Array], tuple[F32Array, U8Array]]:
    (train_images, train_labels), (test_images, test_labels) = load_raw_data()
    train_images = preprocess_data(train_images)
    test_images = preprocess_data(test_images)
    return (train_images, train_labels), (test_images, test_labels)


def elapsed_time(func: Callable[[], Any]) -> float:
    start = time.perf_counter()
    _ = func()
    return time.perf_counter() - start


def ceil_div(m: int, n: int) -> int:
    return -(-m // n)


def shuffle_in_unison(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    indices = tf.random.shuffle(tf.range(len(x)))
    return tf.gather(x, indices), tf.gather(y, indices)


def set_random_seed_for(
    library: Literal["python", "numpy", "tensorflow", "keras"],
    seed: int | None,
) -> None:
    if seed is None:
        return
    match library:
        case "python":
            random.seed(seed)
        case "numpy":
            np.random.seed(seed)
        case "tensorflow":
            tf.random.set_seed(seed)
        case "keras":
            keras.utils.set_random_seed(seed)
        case _:
            raise ValueError(f"unknown library: {library!r}")


def tf_set_log_level(level: int | None = None, *, argv_index: int = 1, default_level: int = 0) -> None:
    """
    Set the TensorFlow C++ backend log level via the `TF_CPP_MIN_LOG_LEVEL` environment variable.

    Levels:
        0: Show all logs (default)
        1: Filter out INFO logs
        2: Filter out INFO and WARNING logs
        3: Filter out INFO, WARNING, and ERROR logs
    """
    if level is None:
        try:
            level = int(sys.argv[argv_index])
        except (IndexError, ValueError):
            level = int(os.getenv("TF_CPP_MIN_LOG_LEVEL", default=str(default_level)))

    level = min(3, max(0, level))
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)


def tf_debug_log(*, enabled: bool = True):
    tf.debugging.set_log_device_placement(enabled)


class SupportsWeights[T](Protocol):
    @property
    def weights(self) -> Iterable[T]: ...


def flatten_weights1[T](layers: Iterable[SupportsWeights[T]]) -> list[T]:
    weights = []
    for layer in layers:
        for w in layer.weights:
            weights.append(w)
    return weights


def flatten_weights2[T](layers: Iterable[SupportsWeights[T]]) -> list[T]:
    weights = []
    for layer in layers:
        weights.extend(layer.weights)  # weights += layer.weights
    return weights


def flatten_weights3[T](layers: Iterable[SupportsWeights[T]]) -> list[T]:
    return [
        w
        for layer in layers
        for w in layer.weights
    ]


def to_multi_hot[ScalarType: np.generic](
    sequences: Sequence[Sequence[int]],
    *,
    dimension: int = 10000,
    dtype: type[ScalarType] = np.float32,
) -> NDArray[ScalarType]:
    """
    Convert sequences of indices into multi-hot encoded vectors.
    `dimension` is the vocabulary size (maximum index + 1).
    """
    vectors = np.zeros((len(sequences), dimension), dtype=dtype)
    for i, sequence in enumerate(sequences):
        for index in sequence:
            vectors[i, index] = 1
    return vectors


vectorize_sequences = to_multi_hot


def to_one_hot[ScalarType: np.generic](
    labels: Sequence[int],
    *,
    dimension: int,
    dtype: type[ScalarType] = np.float32,
) -> NDArray[ScalarType]:
    """
    Convert labels into one-hot encoded vectors.
    `dimension` is the number of classes (maximum label + 1).
    """
    vectors = np.zeros((len(labels), dimension), dtype=dtype)
    for i, label in enumerate(labels):
        vectors[i, label] = 1
    return vectors


to_categorical = to_one_hot

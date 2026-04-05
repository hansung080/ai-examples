from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

N_FEATURES = 3
N_CLASSES = 2
EPOCHS = 100_000
LEARNING_RATE = np.float32(0.05)
CLASSIFICATION_THRESHOLD = np.float32(0.5)

ZERO = np.float32(0.0)
ONE = np.float32(1.0)
TWO = np.float32(2.0)

F32Array: TypeAlias = NDArray[np.float32]
U8Array: TypeAlias = NDArray[np.uint8]
BoolArray: TypeAlias = NDArray[np.bool_]


def _load_raw_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "light_dark_font_training_set.csv"
    if data_path.is_file():
        return pd.read_csv(data_path)
    else:
        # https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv
        # https://bit.ly/3GsNzGt
        return pd.read_csv("https://tinyurl.com/y2qmhfsr")


def preprocess_data(x: U8Array | F32Array) -> F32Array:
    return x / np.float32(255.0)


def load_data(*, seed: int | None = None) -> tuple[tuple[F32Array, U8Array], tuple[F32Array, U8Array]]:
    data = _load_raw_data()
    x_all: F32Array = preprocess_data(data.iloc[:, :-1].to_numpy(dtype=np.float32))
    y_all: U8Array = data.iloc[:, -1].to_numpy(dtype=np.uint8)

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1 / 3, random_state=seed)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    return (x_train, y_train), (x_test, y_test)


def relu(x: F32Array) -> F32Array:
    return np.maximum(ZERO, x)


def d_relu(x: F32Array) -> F32Array:
    return (x > ZERO).astype(np.float32)


def sigmoid(x: F32Array) -> F32Array:
    return ONE / (ONE + np.exp(-x))


def d_sigmoid(x: F32Array) -> F32Array:
    exp_neg_x = np.exp(-x)
    return exp_neg_x / (ONE + exp_neg_x) ** TWO


def elapsed_time(func: Callable[[], Any]) -> float:
    start = time.perf_counter()
    _ = func()
    return time.perf_counter() - start

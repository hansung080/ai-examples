from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

N_EPOCHS = 100_000
LEARNING_RATE = np.float32(0.05)
CLASSIFICATION_THRESHOLD = np.float32(0.5)

F32Array: TypeAlias = NDArray[np.float32]
I64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def read_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "light_dark_font_training_set.csv"
    if data_path.is_file():
        return pd.read_csv(data_path)
    else:
        # https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv
        # https://bit.ly/3GsNzGt
        return pd.read_csv("https://tinyurl.com/y2qmhfsr")


def read_and_split_data() -> tuple[F32Array, F32Array, I64Array, I64Array]:
    data = read_data()
    x_all: F32Array = data.iloc[:, :-1].to_numpy(dtype=np.float32) / np.float32(255.0)
    y_all: I64Array = data.iloc[:, -1].to_numpy(dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1 / 3)
    return x_train, x_test, y_train, y_test


def relu(x: F32Array) -> F32Array:
    return np.maximum(np.float32(0.0), x)


def d_relu(x: F32Array) -> F32Array:
    return (x > np.float32(0.0)).astype(np.float32)


def sigmoid(x: F32Array) -> F32Array:
    one = np.float32(1.0)
    return one / (one + np.exp(-x))


def d_sigmoid(x: F32Array) -> F32Array:
    exp_neg_x = np.exp(-x)
    return exp_neg_x / (np.float32(1.0) + exp_neg_x) ** np.float32(2.0)


def elapsed_time(func: Callable[[], Any]) -> float:
    start = time.perf_counter()
    _ = func()
    return time.perf_counter() - start

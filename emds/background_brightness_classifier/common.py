from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FArray: TypeAlias = NDArray[np.floating]
IArray: TypeAlias = NDArray[np.integer]
BArray: TypeAlias = NDArray[np.bool_]


def read_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "light_dark_font_training_set.csv"
    if data_path.is_file():
        return pd.read_csv(data_path)
    else:
        # https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv
        return pd.read_csv("https://tinyurl.com/y2qmhfsr")


def relu(x: FArray) -> FArray:
    return np.maximum(0, x)


def d_relu(x: FArray) -> FArray:
    return (x > 0).astype(x.dtype)


def sigmoid(x: FArray) -> FArray:
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: FArray) -> FArray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

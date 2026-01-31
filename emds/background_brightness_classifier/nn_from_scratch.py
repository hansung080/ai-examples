#!../../.venv/bin/python3
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

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


all_data: pd.DataFrame = read_data()

x_all: FArray = all_data.iloc[:, 0:3].values / 255.0
y_all: IArray = all_data.iloc[:, -1].values

x_train: FArray
x_test: FArray
y_train: IArray
y_test: IArray

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1 / 3)
n_samples: int = x_train.shape[0]

w1: FArray = np.random.rand(3, 3)
b1: FArray = np.random.rand(3, 1)

w2: FArray = np.random.rand(1, 3)
b2: FArray = np.random.rand(1, 1)


def relu(z: FArray) -> FArray:
    return np.maximum(0, z)


def sigmoid(z: FArray) -> FArray:
    return 1 / (1 + np.exp(-z))


def forward_pass(x: FArray) -> tuple[FArray, FArray, FArray, FArray]:
    z1 = w1 @ x + b1   # (n_features=3, n_samples=449) = (n_out_features=3, n_in_features=3) @ (n_features=3, n_samples=449) + (n_out_features=3, 1)
    a1 = relu(z1)      # (n_features=3, n_samples=449) = relu((n_features=3, n_samples=449))
    z2 = w2 @ a1 + b2  # (n_features=1, n_samples=449) = (n_out_features=1, n_in_features=3) @ (n_features=3, n_samples=449) + (n_out_features=1, 1)
    a2 = sigmoid(z2)   # (n_features=1, n_samples=449) = sigmoid((n_features=1, n_samples=449))
    return z1, a1, z2, a2


y_pred: FArray = forward_pass(x_test.T)[3]
y_hat: IArray = (y_pred >= 0.5).flatten().astype(int)
correct_mask: BArray = np.equal(y_hat, y_test)
accuracy: float = correct_mask.mean()
print(f"accuracy: {accuracy}")

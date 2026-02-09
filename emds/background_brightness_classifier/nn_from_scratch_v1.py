#!../../.venv/bin/python3
from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from common import CLASSIFICATION_THRESHOLD
from common import BArray, FArray, IArray, d_relu, d_sigmoid, read_data, relu, sigmoid

N_EPOCHS = 100_000
LEARNING_RATE = 0.05


class NeuralNetwork:
    def __init__(self) -> None:
        all_data = read_data()
        x_all: FArray = all_data.iloc[:, 0:3].values / 255.0
        y_all: IArray = all_data.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1 / 3)

        self.x_train: FArray = x_train
        self.y_train: IArray = y_train
        self.x_test: FArray = x_test
        self.y_test: IArray = y_test

        self.w1: FArray = np.random.rand(3, 3)
        self.b1: FArray = np.random.rand(1, 3)
        self.w2: FArray = np.random.rand(3, 1)
        self.b2: FArray = np.random.rand(1, 1)

    def forward_prop(self, x: FArray) -> tuple[FArray, FArray, FArray, FArray]:
        z1 = x @ self.w1 + self.b1   # (N, 3) = (N, 3) @ (3, 3) + (1, 3)
        a1 = relu(z1)                # (N, 3) = relu((N, 3))
        z2 = a1 @ self.w2 + self.b2  # (N, 1) = (N, 3) @ (3, 1) + (1, 1)
        a2 = sigmoid(z2)             # (N, 1) = sigmoid((N, 1))
        return z1, a1, z2, a2

    def backward_prop(
        self,
        x: FArray,
        y: IArray,
        z1: FArray,
        a1: FArray,
        z2: FArray,
        a2: FArray,
    ) -> tuple[FArray, FArray, FArray, FArray]:
        dl_da2 = 2 * a2 - 2 * y
        da2_dz2 = d_sigmoid(z2)
        dz2_dw2 = a1
        dz2_db2 = 1
        dz2_da1 = self.w2
        da1_dz1 = d_relu(z1)
        dz1_dw1 = x
        dz1_db1 = 1

        dl_dw2 = dl_da2 @ da2_dz2 @ dz2_dw2    # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_db2 = dl_da2 @ da2_dz2 * dz2_db2    # (1, 1) = (1, 1) @ (1, 1) * 1
        dl_da1 = dl_da2 @ da2_dz2 @ dz2_da1.T  # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_dw1 = dl_da1 @ da1_dz1.T @ dz1_dw1  # (1, 3) = (1, 3) @ (3, 1) @ (1, 3)
        dl_db1 = dl_da1 @ da1_dz1.T * dz1_db1  # (1, 1) = (1, 3) @ (3, 1) * 1
        return cast(
            tuple[FArray, FArray, FArray, FArray],
            (dl_dw1.T, dl_db1, dl_dw2.T, dl_db2),
        )

    def train(self) -> None:
        n_samples = self.x_train.shape[0]
        for epoch in range(N_EPOCHS):
            index = np.random.choice(n_samples, 1, replace=False)
            x_sample = self.x_train[index]
            y_sample = self.y_train[index]

            z1, a1, z2, a2 = self.forward_prop(x_sample)

            dl_dw1, dl_db1, dl_dw2, dl_db2 = self.backward_prop(x_sample, y_sample, z1, a1, z2, a2)

            self.w1 -= LEARNING_RATE * dl_dw1
            self.b1 -= LEARNING_RATE * dl_db1
            self.w2 -= LEARNING_RATE * dl_dw2
            self.b2 -= LEARNING_RATE * dl_db2

    def evaluate(self) -> float:
        y_pred: FArray = self.forward_prop(self.x_test)[3]
        y_hat: IArray = (y_pred >= CLASSIFICATION_THRESHOLD).flatten().astype(int)
        correct_mask: BArray = np.equal(y_hat, self.y_test)
        accuracy = correct_mask.mean()
        return accuracy


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.train()
    print(f"accuracy: {nn.evaluate():.2%}")

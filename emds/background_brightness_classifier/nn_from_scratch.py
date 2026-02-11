#!../../.venv/bin/python3
from __future__ import annotations

from typing import cast

import numpy as np

from common import CLASSIFICATION_THRESHOLD
from common import BArray, FArray, IArray, d_relu, d_sigmoid, elapsed_time, read_and_split_data, relu, sigmoid

N_EPOCHS = 100_000
LEARNING_RATE = np.float32(0.05)


class NeuralNetwork:
    def __init__(self, *, seed: int | None = None) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = read_and_split_data()

        self.rng = np.random.default_rng(seed)
        self.w1: FArray = self.rng.random((3, 3), dtype=np.float32)
        self.b1: FArray = self.rng.random((1, 3), dtype=np.float32)
        self.w2: FArray = self.rng.random((3, 1), dtype=np.float32)
        self.b2: FArray = self.rng.random((1, 1), dtype=np.float32)

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
        one = np.float32(1.0)
        two = np.float32(2.0)

        dl_da2 = two * a2 - two * y.astype(np.float32)
        da2_dz2 = d_sigmoid(z2)
        dz2_dw2 = a1
        dz2_db2 = one
        dz2_da1 = self.w2
        da1_dz1 = d_relu(z1)
        dz1_dw1 = x
        dz1_db1 = one

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
            index = self.rng.choice(n_samples, 1, replace=False)
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
        y_hat: IArray = (y_pred >= CLASSIFICATION_THRESHOLD).flatten().astype(np.int64)
        correct_mask: BArray = np.equal(y_hat, self.y_test)
        accuracy: np.float32 = correct_mask.mean()
        return float(accuracy)

    def predict(self, r: int, g: int, b: int) -> float:
        x: FArray = np.array([[r, g, b]], dtype=np.float32) / np.float32(255.0)
        y_pred: FArray = self.forward_prop(x)[3]
        return float(y_pred[0, 0])


if __name__ == "__main__":
    nn = NeuralNetwork()
    elapsed = elapsed_time(nn.train)
    print(f"train time: {elapsed:.2f}s, accuracy: {nn.evaluate():.2%}")

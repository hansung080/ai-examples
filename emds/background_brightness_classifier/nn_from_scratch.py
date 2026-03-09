#!../../.venv/bin/python
from __future__ import annotations

from typing import cast

import numpy as np

from common import CLASSIFICATION_THRESHOLD, LEARNING_RATE, N_EPOCHS
from common import BoolArray, F32Array, I64Array, d_relu, d_sigmoid, elapsed_time, read_and_split_data, relu, sigmoid


class NeuralNetwork:
    def __init__(self, *, seed: int | None = None) -> None:
        self._x_train, self._x_test, self._y_train, self._y_test = read_and_split_data()

        self._rng = np.random.default_rng(seed)
        self._w1: F32Array = self._rng.random((3, 3), dtype=np.float32)
        self._b1: F32Array = self._rng.random((1, 3), dtype=np.float32)
        self._w2: F32Array = self._rng.random((3, 1), dtype=np.float32)
        self._b2: F32Array = self._rng.random((1, 1), dtype=np.float32)

    def weights(self) -> list[F32Array]:
        return [self._w1, self._w2]

    def biases(self) -> list[F32Array]:
        return [self._b1, self._b2]

    def _forward_prop(self, x: F32Array) -> tuple[F32Array, F32Array, F32Array, F32Array]:
        z1 = x @ self._w1 + self._b1   # (N, 3) = (N, 3) @ (3, 3) + (1, 3)
        a1 = relu(z1)                  # (N, 3) = relu((N, 3))
        z2 = a1 @ self._w2 + self._b2  # (N, 1) = (N, 3) @ (3, 1) + (1, 1)
        a2 = sigmoid(z2)               # (N, 1) = sigmoid((N, 1))
        return z1, a1, z2, a2

    def _backward_prop(
        self,
        x: F32Array,
        y: I64Array,
        z1: F32Array,
        a1: F32Array,
        z2: F32Array,
        a2: F32Array,
    ) -> tuple[F32Array, F32Array, F32Array, F32Array]:
        one = np.float32(1.0)
        two = np.float32(2.0)

        dl_da2 = two * a2 - two * y.astype(np.float32)
        da2_dz2 = d_sigmoid(z2)
        dz2_dw2 = a1
        dz2_db2 = one
        dz2_da1 = self._w2
        da1_dz1 = d_relu(z1)
        dz1_dw1 = x
        dz1_db1 = one

        dl_dw2 = dl_da2 @ da2_dz2 @ dz2_dw2    # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_db2 = dl_da2 @ da2_dz2 * dz2_db2    # (1, 1) = (1, 1) @ (1, 1) * 1
        dl_da1 = dl_da2 @ da2_dz2 @ dz2_da1.T  # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_dw1 = dl_da1 @ da1_dz1.T @ dz1_dw1  # (1, 3) = (1, 3) @ (3, 1) @ (1, 3)
        dl_db1 = dl_da1 @ da1_dz1.T * dz1_db1  # (1, 1) = (1, 3) @ (3, 1) * 1
        return cast(
            tuple[F32Array, F32Array, F32Array, F32Array],
            (dl_dw1.T, dl_db1, dl_dw2.T, dl_db2),
        )

    def train(self) -> None:
        n_samples = self._x_train.shape[0]

        for epoch in range(N_EPOCHS):
            index = self._rng.choice(n_samples, 1, replace=False)
            x_sample = self._x_train[index]
            y_sample = self._y_train[index]

            z1, a1, z2, a2 = self._forward_prop(x_sample)

            dl_dw1, dl_db1, dl_dw2, dl_db2 = self._backward_prop(x_sample, y_sample, z1, a1, z2, a2)

            self._w1 -= LEARNING_RATE * dl_dw1
            self._b1 -= LEARNING_RATE * dl_db1
            self._w2 -= LEARNING_RATE * dl_dw2
            self._b2 -= LEARNING_RATE * dl_db2

    def evaluate(self) -> float:
        y_pred: F32Array = self._forward_prop(self._x_test)[3]
        y_hat: I64Array = (y_pred >= CLASSIFICATION_THRESHOLD).flatten().astype(np.int64)
        correct_mask: BoolArray = np.equal(y_hat, self._y_test)
        accuracy: np.float32 = correct_mask.mean()
        return float(accuracy)

    def predict(self, r: int, g: int, b: int) -> float:
        x: F32Array = np.array([[r, g, b]], dtype=np.float32) / np.float32(255.0)
        y_pred: F32Array = self._forward_prop(x)[3]
        return float(y_pred[0, 0])


if __name__ == "__main__":
    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    print(f"train time: {train_time:.2f}s, accuracy: {nn.evaluate():.2%}")

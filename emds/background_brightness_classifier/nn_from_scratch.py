#!../../.venv/bin/python
from __future__ import annotations

from typing import cast

import numpy as np

from common import CLASSIFICATION_THRESHOLD, LEARNING_RATE, N_CLASSES, N_EPOCHS, N_FEATURES
from common import BoolArray, F32Array, U8Array
from common import d_relu, d_sigmoid, elapsed_time, load_data, preprocess_data, relu, sigmoid
from nn_protocol import Background, Evaluation


class NeuralNetwork:
    def __init__(self, *, seed: int | None = None) -> None:
        (self._x_train, self._y_train), (self._x_test, self._y_test) = load_data()

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
        y: U8Array,
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

        for _ in range(N_EPOCHS):
            index = self._rng.choice(n_samples, 1, replace=False)
            x_sample = self._x_train[index]
            y_sample = self._y_train[index]

            z1, a1, z2, a2 = self._forward_prop(x_sample)

            dl_dw1, dl_db1, dl_dw2, dl_db2 = self._backward_prop(x_sample, y_sample, z1, a1, z2, a2)

            self._w1 -= LEARNING_RATE * dl_dw1
            self._b1 -= LEARNING_RATE * dl_db1
            self._w2 -= LEARNING_RATE * dl_dw2
            self._b2 -= LEARNING_RATE * dl_db2

    def evaluate(self) -> Evaluation:
        y_proba: F32Array = self._forward_prop(self._x_test)[3]
        y_pred: U8Array = (y_proba >= CLASSIFICATION_THRESHOLD).flatten().astype(np.uint8)
        correct_mask: BoolArray = np.equal(y_pred, self._y_test)
        accuracy: np.float32 = correct_mask.mean()
        return Evaluation(float(accuracy))

    def predict_proba(self, inputs: U8Array | F32Array) -> F32Array:
        assert inputs.ndim == 2 and inputs.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(inputs)
        y_proba: F32Array = self._forward_prop(x)[3]

        # `np.concatenate((np.float32(1.0) - y_proba, y_proba), axis=1)` can be used instead of `np.hstack(...)`.
        y_proba: F32Array = np.hstack((np.float32(1.0) - y_proba, y_proba))
        assert y_proba.shape == (inputs.shape[0], N_CLASSES)
        return y_proba

    def predict(self, inputs: U8Array | F32Array) -> U8Array:
        y_proba: F32Array = self.predict_proba(inputs)

        # Binary classification:     sigmoid -> threshold
        # Multilabel classification: sigmoid -> threshold
        # Multiclass classification: softmax -> argmax: `y_proba.argmax(axis=1).astype(np.uint8)`
        y_pred: U8Array = (y_proba[:, 1] >= CLASSIFICATION_THRESHOLD).astype(np.uint8)
        assert y_pred.shape == (inputs.shape[0],)
        return y_pred

    def predict_one(self, r: int, g: int, b: int) -> Background:
        y_pred: U8Array = self.predict(np.array([[r, g, b]], dtype=np.float32))
        return Background(y_pred[0])


if __name__ == "__main__":
    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    print(f"train time: {train_time:.2f}s, accuracy: {nn.evaluate().accuracy:.2%}")

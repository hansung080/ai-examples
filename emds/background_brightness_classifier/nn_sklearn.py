#!../../.venv/bin/python
from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from sklearn.neural_network import MLPClassifier

from common import LEARNING_RATE, N_EPOCHS
from common import F32Array, elapsed_time, read_and_split_data


class NeuralNetwork:
    def __init__(self, *, random_state: int | RandomState | None = None) -> None:
        self._x_train, self._x_test, self._y_train, self._y_test = read_and_split_data()

        self._nn = MLPClassifier(
            hidden_layer_sizes=(3,),
            activation="relu",
            solver="sgd",
            max_iter=N_EPOCHS,
            learning_rate_init=float(LEARNING_RATE),
            random_state=random_state,
        )

    def weights(self) -> list[F32Array] | None:
        try:
            return self._nn.coefs_
        except AttributeError:
            return None

    def biases(self) -> list[F32Array] | None:
        try:
            return self._nn.intercepts_
        except AttributeError:
            return None

    def train(self) -> None:
        self._nn.fit(self._x_train, self._y_train)

    def evaluate(self) -> float:
        return self._nn.score(self._x_test, self._y_test)

    def predict(self, r: int, g: int, b: int) -> float:
        x: F32Array = np.array([[r, g, b]], dtype=np.float32) / np.float32(255.0)
        y_pred: F32Array = self._nn.predict_proba(x)
        return float(y_pred[0, 1])


if __name__ == "__main__":
    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    print(f"train time: {train_time:.2f}s, accuracy: {nn.evaluate():.2%}")

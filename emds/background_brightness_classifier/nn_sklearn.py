#!../../.venv/bin/python
from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from sklearn.neural_network import MLPClassifier

from common import LEARNING_RATE, N_CLASSES, N_EPOCHS, N_FEATURES
from common import F32Array, U8Array, elapsed_time, load_data, preprocess_data
from nn_protocol import Background, Evaluation


class NeuralNetwork:
    def __init__(self, *, random_state: int | RandomState | None = None) -> None:
        (self._x_train, self._y_train), (self._x_test, self._y_test) = load_data()

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

    def evaluate(self) -> Evaluation:
        accuracy: float = self._nn.score(self._x_test, self._y_test)
        return Evaluation(accuracy)

    def predict_proba(self, inputs: U8Array | F32Array) -> F32Array:
        assert inputs.ndim == 2 and inputs.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(inputs)
        y_proba: F32Array = self._nn.predict_proba(x)
        assert y_proba.shape == (inputs.shape[0], N_CLASSES)
        return y_proba

    def predict(self, inputs: U8Array | F32Array) -> U8Array:
        assert inputs.ndim == 2 and inputs.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(inputs)
        y_pred: U8Array = self._nn.predict(x)
        assert y_pred.shape == (inputs.shape[0],)
        return y_pred

    def predict_one(self, r: int, g: int, b: int) -> Background:
        y_pred: U8Array = self.predict(np.array([[r, g, b]], dtype=np.float32))
        return Background(y_pred[0])


if __name__ == "__main__":
    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    print(f"train time: {train_time:.2f}s, accuracy: {nn.evaluate().accuracy:.2%}")

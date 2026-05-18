#!../../.venv/bin/python
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from common import EPOCHS, LEARNING_RATE, N_CLASSES, N_FEATURES
from common import F32Array, U8Array, elapsed_time, load_data, preprocess_data
from nn_protocol import Background, Evaluation


class NeuralNetwork:
    def __init__(self, *, seed: int | None = None) -> None:
        (self._x_train, self._y_train), (self._x_test, self._y_test) = load_data(seed=seed)

        self._model = MLPClassifier(
            hidden_layer_sizes=(3,),
            activation="relu",
            solver="sgd",  # mini-batch SGD with batch size `min(n_samples, 200)`
            learning_rate_init=float(LEARNING_RATE),
            max_iter=EPOCHS,
            random_state=seed,
        )

    @property
    def weights(self) -> list[F32Array] | None:
        try:
            return [
                x
                for w, b in zip(self._model.coefs_, self._model.intercepts_, strict=True)
                for x in (w, b)
            ]
        except AttributeError:
            return None

    # In this mini-batch SGD, 500000 (= 100000 * ceil(896 / 200)) weight updates are performed.
    def train(self) -> None:
        self._model.fit(self._x_train, self._y_train)

    def evaluate(self) -> Evaluation:
        accuracy: float = self._model.score(self._x_test, self._y_test)
        return Evaluation(accuracy)

    def predict_probs(self, colors: U8Array | F32Array) -> F32Array:
        assert colors.ndim == 2 and colors.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(colors)
        y_prob: F32Array = self._model.predict_proba(x)
        assert y_prob.shape == (colors.shape[0], N_CLASSES)
        return y_prob

    def predict(self, colors: U8Array | F32Array) -> U8Array:
        assert colors.ndim == 2 and colors.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(colors)
        y_pred: U8Array = self._model.predict(x)
        assert y_pred.shape == (colors.shape[0],)
        return y_pred

    def predict_one(self, r: int, g: int, b: int) -> Background:
        y_pred: U8Array = self.predict(np.array([[r, g, b]], dtype=np.float32))
        return Background(y_pred[0])


def _run() -> None:
    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    evaluation = nn.evaluate()
    print(f"TRAIN TIME: {train_time:.2f}s, ACCURACY: {evaluation.accuracy:.2%}")


if __name__ == "__main__":
    _run()

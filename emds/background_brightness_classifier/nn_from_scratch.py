#!../../.venv/bin/python
from __future__ import annotations

import numpy as np

from common import CLASSIFICATION_THRESHOLD, EPOCHS, LEARNING_RATE, N_CLASSES, N_FEATURES
from common import ONE, TWO
from common import BoolArray, F32Array, U8Array
from common import d_relu, d_sigmoid, elapsed_time, load_data, preprocess_data, relu, sigmoid
from nn_protocol import Background, Evaluation

STEPS = EPOCHS


class NeuralNetwork:
    def __init__(self, *, seed: int | None = None) -> None:
        (self._x_train, self._y_train), (self._x_test, self._y_test) = load_data(seed=seed)

        self._rng = np.random.default_rng(seed)

        self._w1: F32Array = self._rng.random((3, 3), dtype=np.float32)
        self._b1: F32Array = self._rng.random((1, 3), dtype=np.float32)
        self._w2: F32Array = self._rng.random((3, 1), dtype=np.float32)
        self._b2: F32Array = self._rng.random((1, 1), dtype=np.float32)

    @property
    def weights(self) -> list[F32Array]:
        return [self._w1, self._b1, self._w2, self._b2]

    def _forward_pass(self, x: F32Array) -> tuple[F32Array, F32Array, F32Array, F32Array]:
        z1 = x @ self._w1 + self._b1   # (N, 3) = (N, 3) @ (3, 3) + (1, 3)
        a1 = relu(z1)                  # (N, 3) = relu((N, 3))
        z2 = a1 @ self._w2 + self._b2  # (N, 1) = (N, 3) @ (3, 1) + (1, 1)
        a2 = sigmoid(z2)               # (N, 1) = sigmoid((N, 1))
        return z1, a1, z2, a2

    def _backward_pass(
        self,
        x: F32Array,
        y: F32Array,
        z1: F32Array,
        a1: F32Array,
        z2: F32Array,
        a2: F32Array,
    ) -> tuple[F32Array, F32Array, F32Array, F32Array]:
        dl_da2 = TWO * a2 - TWO * y
        da2_dz2 = d_sigmoid(z2)
        dz2_dw2 = a1
        dz2_db2 = ONE
        dz2_da1 = self._w2
        da1_dz1 = d_relu(z1)
        dz1_dw1 = x
        dz1_db1 = ONE

        dl_dw2 = dl_da2 @ da2_dz2 @ dz2_dw2    # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_db2 = dl_da2 @ da2_dz2 * dz2_db2    # (1, 1) = (1, 1) @ (1, 1) * 1
        dl_da1 = dl_da2 @ da2_dz2 @ dz2_da1.T  # (1, 3) = (1, 1) @ (1, 1) @ (1, 3)
        dl_dw1 = dl_da1 @ da1_dz1.T @ dz1_dw1  # (1, 3) = (1, 3) @ (3, 1) @ (1, 3)
        dl_db1 = dl_da1 @ da1_dz1.T * dz1_db1  # (1, 1) = (1, 3) @ (3, 1) * 1
        return dl_dw1.T, dl_db1, dl_dw2.T, dl_db2

    # === [Stochastic] Gradient Descent ===
    #
    # Online SGD (batch_size = 1):
    #   - Each sample updates the weights using its gradient.
    #   - Weight updates: epochs * n_samples (without replacement), steps (with replacement)
    #
    # Mini-batch SGD (1 < batch_size < n_samples):
    #   - Each batch updates the weights using the gradient averaged over the batch.
    #   - Weight updates: epochs * n_batches -> epochs * ceil(n_samples / batch_size)
    #
    # Full-batch GD (batch_size = n_samples):
    #   - Each epoch updates the weights using the gradient averaged over the entire dataset.
    #   - Weight updates: epochs
    #
    # Notes:
    #   - An epoch is a full pass through the entire dataset.
    #   - A batch is a subset of the dataset used for one step of training.
    #   - A step consists of a forward pass, a backward pass, and a weight update,
    #     and uses a single sample for online SGD, a batch for mini-batch SGD, and the entire dataset for full-batch GD.

    # In this online SGD with replacement, 100000 weight updates are performed.
    def train(self) -> None:
        x_train = self._x_train
        y_train = self._y_train.astype(np.float32)
        n_samples = x_train.shape[0]

        for _ in range(STEPS):
            index = self._rng.choice(n_samples, 1, replace=False)
            x_sample = x_train[index]
            y_sample = y_train[index]

            z1, a1, z2, a2 = self._forward_pass(x_sample)

            dl_dw1, dl_db1, dl_dw2, dl_db2 = self._backward_pass(x_sample, y_sample, z1, a1, z2, a2)

            self._w1 -= LEARNING_RATE * dl_dw1
            self._b1 -= LEARNING_RATE * dl_db1
            self._w2 -= LEARNING_RATE * dl_dw2
            self._b2 -= LEARNING_RATE * dl_db2

    def evaluate(self) -> Evaluation:
        y_prob: F32Array = self._forward_pass(self._x_test)[3]
        y_pred: U8Array = (y_prob >= CLASSIFICATION_THRESHOLD).flatten().astype(np.uint8)
        correct_mask: BoolArray = np.equal(y_pred, self._y_test)
        accuracy: np.float32 = np.mean(correct_mask)
        return Evaluation(float(accuracy))

    def predict_probs(self, colors: U8Array | F32Array) -> F32Array:
        assert colors.ndim == 2 and colors.shape[1] == N_FEATURES
        x: F32Array = preprocess_data(colors)
        y_prob: F32Array = self._forward_pass(x)[3]

        # `np.concatenate((ONE - y_prob, y_prob), axis=1)` can be an alternative to `np.hstack(...)`.
        y_prob = np.hstack((ONE - y_prob, y_prob))
        assert y_prob.shape == (colors.shape[0], N_CLASSES)
        return y_prob

    def predict(self, colors: U8Array | F32Array) -> U8Array:
        y_prob: F32Array = self.predict_probs(colors)

        # binary classification:      sigmoid -> threshold
        # multiclass classification:  softmax -> argmax: `np.argmax(y_prob, axis=1).astype(np.uint8)`
        # multi-label classification: sigmoid -> threshold
        y_pred: U8Array = (y_prob[:, 1] >= CLASSIFICATION_THRESHOLD).astype(np.uint8)
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

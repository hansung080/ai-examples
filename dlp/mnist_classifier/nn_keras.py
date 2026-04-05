#!../../.venv/bin/python
from __future__ import annotations

import keras
import numpy as np
from keras import layers

from common import BATCH_SIZE, EPOCHS, HIDDEN_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, N_CLASSES
from common import F32Array, U8Array
from common import elapsed_time, load_data, load_raw_data, preprocess_data, set_random_seed_for, tf_set_log_level
from nn_protocol import Digit, Evaluation


class NeuralNetwork:
    def __init__(self) -> None:
        (self._train_images, self._train_labels), (self._test_images, self._test_labels) = load_data()

        self._model = keras.Sequential([
            layers.Dense(units=HIDDEN_SIZE, activation="relu"),
            layers.Dense(units=N_CLASSES, activation="softmax"),
        ])

        self._model.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    @property
    def weights(self) -> list[F32Array]:
        return self._model.get_weights()

    # In this mini-batch SGD, 2345 (= 5 * ceil(60000 / 128)) weight updates are performed.
    def train(self) -> None:
        self._model.fit(
            self._train_images,
            self._train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )

    def evaluate(self) -> Evaluation:
        loss, accuracy = self._model.evaluate(self._test_images, self._test_labels)
        return Evaluation(loss, accuracy)

    def predict_proba(self, images: U8Array) -> F32Array:
        assert images.ndim == 3 and images.shape[1] == IMAGE_HEIGHT and images.shape[2] == IMAGE_WIDTH
        images: F32Array = preprocess_data(images)
        probabilities: F32Array = self._model.predict(images)
        assert probabilities.shape == (images.shape[0], N_CLASSES)
        return probabilities

    def predict(self, images: U8Array) -> U8Array:
        probabilities: F32Array = self.predict_proba(images)
        predictions: U8Array = np.argmax(probabilities, axis=1).astype(np.uint8)
        assert predictions.shape == (images.shape[0],)
        return predictions

    def predict_one(self, image: U8Array) -> Digit:
        assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
        predictions: U8Array = self.predict(image[np.newaxis, ...])
        return Digit(predictions[0])


if __name__ == "__main__":
    set_random_seed_for(library="keras", seed=None)
    tf_set_log_level(argv_index=1, default_level=1)

    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    evaluation = nn.evaluate()
    print(f"TRAIN TIME: {train_time:.2f}s, ACCURACY: {evaluation.accuracy:.2%}, LOSS: {evaluation.loss:.4f}")

    _, (test_images, test_labels) = load_raw_data()
    assert nn.predict_one(test_images[0]) == test_labels[0]

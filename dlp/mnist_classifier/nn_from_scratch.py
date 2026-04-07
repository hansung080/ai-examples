#!../../.venv/bin/python
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Callable, Literal, cast

import keras
import numpy as np
import tensorflow as tf

from common import BATCH_SIZE, EPOCHS, HIDDEN_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, N_CLASSES, N_FEATURES
from common import F32Array, U8Array
from common import ceil_div, elapsed_time, load_data, load_raw_data, preprocess_data, set_random_seed_for
from common import shuffle_in_unison, tf_set_log_level
from nn_protocol import Digit, Evaluation


class NaiveDense:
    def __init__(self, input_size: int, output_size: int, activation: Callable[[tf.Tensor], tf.Tensor]) -> None:
        self._activation = activation

        self._W = tf.Variable(
            tf.random.uniform(
                (input_size, output_size),
                minval=0,
                maxval=0.1,
                dtype=tf.float32,
            ),
        )

        self._b = tf.Variable(
            tf.zeros(
                (output_size,),
                dtype=tf.float32,
            ),
        )

    @property
    def weights(self) -> list[tf.Variable]:
        return [self._W, self._b]

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._activation(inputs @ self._W + self._b)


class NaiveSequential:
    def __init__(self, layers: Sequence[NaiveDense]) -> None:
        self._layers = list(layers)
        self._optimizer = None

        self._weights: list[tf.Variable] = []
        for layer in self._layers:
            self._weights.extend(layer.weights)

    def compile(self, *, optimizer: Literal["sgd", "sgd_momentum", "rmsprop"] | None = None) -> None:
        match optimizer:
            case "sgd":
                self._optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
            case "sgd_momentum":
                self._optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE * 0.7, momentum=0.9)
            case "rmsprop":
                self._optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
            case None:
                self._optimizer = None
            case _:
                raise ValueError(f"unknown optimizer: {optimizer!r}")

    @property
    def weights(self) -> list[tf.Variable]:
        return self._weights

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    @staticmethod
    def compute_loss(targets: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        per_sample_losses = keras.losses.sparse_categorical_crossentropy(targets, outputs)
        return tf.reduce_mean(per_sample_losses)

    def _update_weights(self, gradients: Sequence[tf.Tensor | None]) -> None:
        if self._optimizer is not None:
            self._optimizer.apply_gradients(
                (g, w)
                for g, w in zip(gradients, self._weights, strict=True)
                if g is not None
            )
        else:
            for w, g in zip(self._weights, gradients, strict=True):
                if g is not None:
                    w.assign_sub(g * LEARNING_RATE)

    def _one_training_step(self, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.compute_loss(targets, outputs)
        gradients = tape.gradient(loss, self._weights)
        self._update_weights(gradients)
        return loss

    def fit(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        *,
        batch_size: int = 128,
        epochs: int = 1,
        verbose: bool = False,
        shuffle: bool = True,
    ) -> None:
        batch_size = min(len(inputs), max(1, batch_size))
        for epoch in range(epochs):
            inputs_epoch, targets_epoch = shuffle_in_unison(inputs, targets) if shuffle else (inputs, targets)
            batches = BatchIterator(inputs_epoch, targets_epoch, batch_size)
            batch, loss = -1, 0.0
            for batch, (inputs_batch, targets_batch) in enumerate(batches):
                loss = self._one_training_step(inputs_batch, targets_batch)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}, batch {batch + 1}/{len(batches)} => loss: {loss:.4f}")


class BatchIterator(Iterator[tuple[tf.Tensor, tf.Tensor]]):
    def __init__(self, inputs: tf.Tensor, targets: tf.Tensor, batch_size: int):
        assert len(inputs) == len(targets)
        assert batch_size > 0

        self._inputs = inputs
        self._targets = targets
        self._index = 0
        self._batch_size = batch_size
        self._n_batches = ceil_div(len(inputs), batch_size)

    def __iter__(self) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
        return self

    def __next__(self) -> tuple[tf.Tensor, tf.Tensor]:
        if self._index >= len(self._inputs):
            raise StopIteration
        end_index = self._index + self._batch_size
        inputs = self._inputs[self._index:end_index]
        targets = self._targets[self._index:end_index]
        self._index = end_index
        return inputs, targets

    def __len__(self) -> int:
        return self._n_batches


class NeuralNetwork:
    def __init__(self) -> None:
        (self._train_images, self._train_labels), (self._test_images, self._test_labels) = load_data()

        self._model = NaiveSequential([
            NaiveDense(input_size=N_FEATURES, output_size=HIDDEN_SIZE, activation=tf.nn.relu),
            NaiveDense(input_size=HIDDEN_SIZE, output_size=N_CLASSES, activation=tf.nn.softmax),
        ])

        self._model.compile(optimizer="rmsprop")

    @property
    def weights(self) -> list[F32Array]:
        return [cast(tf.Tensor, w).numpy() for w in self._model.weights]

    # In this mini-batch SGD, 2345 (= 5 * ceil(60000 / 128)) weight updates are performed.
    def train(self) -> None:
        self._model.fit(
            tf.convert_to_tensor(self._train_images),
            tf.convert_to_tensor(self._train_labels, dtype=tf.int32),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=True,
        )

    def evaluate(self) -> Evaluation:
        inputs = tf.convert_to_tensor(self._test_images)
        targets = tf.convert_to_tensor(self._test_labels, dtype=tf.int32)

        probs = self._model(inputs)
        loss = self._model.compute_loss(targets, probs)
        preds = tf.argmax(probs, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(preds == targets, tf.float32))
        return Evaluation(float(loss.numpy()), float(accuracy.numpy()))

    def predict_probs(self, images: U8Array) -> F32Array:
        assert images.ndim == 3 and images.shape[1] == IMAGE_HEIGHT and images.shape[2] == IMAGE_WIDTH
        images: F32Array = preprocess_data(images)
        probs: tf.Tensor = self._model(tf.convert_to_tensor(images))
        assert probs.shape == (images.shape[0], N_CLASSES)
        return probs.numpy()

    def predict(self, images: U8Array) -> U8Array:
        probs: F32Array = self.predict_probs(images)
        preds: U8Array = np.argmax(probs, axis=1).astype(np.uint8)
        assert preds.shape == (images.shape[0],)
        return preds

    def predict_one(self, image: U8Array) -> Digit:
        assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
        # `image.reshape((1, image.shape[0], image.shape[1]))` or `np.expand_dims(image, axis=0)` can be an alternative.
        preds: U8Array = self.predict(image[np.newaxis, ...])  # or `image[None, :, :]`
        return Digit(preds[0])


if __name__ == "__main__":
    set_random_seed_for(library="tensorflow", seed=None)
    tf_set_log_level(argv_index=1, default_level=1)

    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    evaluation = nn.evaluate()
    print(f"TRAIN TIME: {train_time:.2f}s, ACCURACY: {evaluation.accuracy:.2%}, LOSS: {evaluation.loss:.4f}")

    _, (test_images, test_labels) = load_raw_data()
    assert nn.predict_one(test_images[0]) == test_labels[0]

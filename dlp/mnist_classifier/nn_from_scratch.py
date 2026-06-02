#!../../.venv/bin/python
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, cast, runtime_checkable

import keras  # type: ignore[import-untyped]
import numpy as np
import tensorflow as tf

from common import BATCH_SIZE, EPOCHS, HIDDEN_LAYER_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, N_CLASSES
from common import F32Array, U8Array
from common import ceil_div, elapsed_time, flatten_weights3 as flatten_weights, load_data, load_raw_data
from common import preprocess_data, set_random_seed_for, shuffle_in_unison, tf_set_log_level, to_one_hot, typed
from nn_protocol import Digit, Evaluation

# Type aliases do not support runtime checks, so use a @runtime_checkable protocol instead.
# ```
# type ActivationFn = Callable[[tf.Tensor], tf.Tensor]
# ```


@runtime_checkable  # checks only method existence, not its signature
class ActivationFn(Protocol):
    def __call__(self, x: tf.Tensor, /) -> tf.Tensor: ...


type ActivationLike = (
    Literal["relu", "sigmoid", "softmax"]
    | ActivationFn
)

type OptimizerLike = (
    Literal["sgd", "sgd_momentum", "rmsprop"]
    | keras.optimizers.Optimizer
    | type[keras.optimizers.Optimizer]
)


@runtime_checkable
class LossFn(Protocol):
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, /) -> tf.Tensor: ...


type LossLike = (
    Literal[
        "binary_crossentropy",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_error",
        "mean_absolute_error",
    ]
    | LossFn
)


class NaiveDense:
    def __init__(self, units: int, *, activation: ActivationLike | None = None) -> None:
        self._units = units
        self._activation_fn: ActivationFn | None = self._get_activation_fn(activation)
        self._W: tf.Variable | None = None
        self._b: tf.Variable | None = None
        self._built = False

    @staticmethod
    def _get_activation_fn(activation: ActivationLike | None) -> ActivationFn | None:
        match activation:
            case "relu":
                return cast(ActivationFn, tf.nn.relu)
            case "sigmoid":
                return cast(ActivationFn, tf.nn.sigmoid)
            case "softmax":
                return cast(ActivationFn, tf.nn.softmax)
            case ActivationFn() as activation_fn:
                return activation_fn
            case None:
                return None
            case _:
                raise ValueError(f"unknown activation: {activation!r}")

    @property
    def weights(self) -> list[tf.Variable]:
        if not self._built:
            raise ValueError("layer not built")
        assert self._W is not None and self._b is not None
        return [self._W, self._b]

    def build(self, input_shape: Sequence[int | None] | tf.TensorShape) -> None:
        if input_shape[-1] is None:
            raise ValueError("input_shape[-1] cannot be None")

        input_dim = int(input_shape[-1])

        self._W = tf.Variable(
            tf.random.uniform(
                (input_dim, self._units),  # (input_size, output_size)
                minval=0,
                maxval=0.1,
                dtype=tf.float32,
            ),
        )

        self._b = tf.Variable(
            tf.zeros(
                (self._units,),  # (output_size,)
                dtype=tf.float32,
            ),
        )

        self._built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert self._W is not None and self._b is not None
        outputs = inputs @ self._W + self._b
        if self._activation_fn is not None:
            outputs = self._activation_fn(outputs)
        return outputs

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self._built:
            self.build(inputs.shape)
        return self.call(inputs)


class NaiveSequential:
    def __init__(self, layers: Sequence[NaiveDense]) -> None:
        self._layers = list(layers)
        self._weights: list[tf.Variable] | None = None
        self._optimizer: keras.optimizers.Optimizer | None = None
        self._loss_fn: LossFn | None = None

    @property
    def weights(self) -> list[tf.Variable]:
        if self._weights is None:
            self._weights = flatten_weights(self._layers)
        return self._weights

    @property
    def loss_fn(self) -> LossFn:
        if self._loss_fn is None:
            raise ValueError("loss function not provided")
        return self._loss_fn

    def compile(self, *, optimizer: OptimizerLike | None = "rmsprop", loss: LossLike | None = None) -> None:
        self._optimizer = self._get_optimizer(optimizer)
        self._loss_fn = self._get_loss_fn(loss)

    @staticmethod
    def _get_optimizer(optimizer: OptimizerLike | None) -> keras.optimizers.Optimizer | None:
        match optimizer:
            case "sgd":
                return keras.optimizers.SGD(learning_rate=LEARNING_RATE)
            case "sgd_momentum":
                return keras.optimizers.SGD(learning_rate=LEARNING_RATE * 0.7, momentum=0.9)
            case "rmsprop":
                return keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
            case keras.optimizers.Optimizer() as opt:
                return opt
            case type() as opt_cls if issubclass(opt_cls, keras.optimizers.Optimizer):
                return opt_cls()
            case None:
                return None
            case _:
                raise ValueError(f"unknown optimizer: {optimizer!r}")

    @staticmethod
    def _get_loss_fn(loss: LossLike | None) -> LossFn | None:
        match loss:
            case "binary_crossentropy":
                return cast(LossFn, keras.losses.binary_crossentropy)
            case "categorical_crossentropy":
                return cast(LossFn, keras.losses.categorical_crossentropy)
            case "sparse_categorical_crossentropy":
                return cast(LossFn, keras.losses.sparse_categorical_crossentropy)
            case "mean_squared_error":
                return cast(LossFn, keras.losses.mean_squared_error)
            case "mean_absolute_error":
                return cast(LossFn, keras.losses.mean_absolute_error)
            case LossFn() as loss_fn:
                return loss_fn
            case None:
                return None
            case _:
                raise ValueError(f"unknown activation: {loss!r}")

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    def compute_loss(self, targets: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        per_sample_losses = self.loss_fn(targets, outputs)
        return tf.reduce_mean(per_sample_losses)

    def _update_weights(self, gradients: Sequence[tf.Tensor | None]) -> None:
        if self._optimizer is not None:
            self._optimizer.apply_gradients(
                (g, w)
                for g, w in zip(gradients, self.weights, strict=True)
                if g is not None
            )
        else:
            for w, g in zip(self.weights, gradients, strict=True):
                if g is not None:
                    w.assign_sub(g * LEARNING_RATE)

    @typed(tf.function)  # or @tf_typed_function switches from eager execution to graph execution for performance.
    def _train_step(self, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.compute_loss(targets, outputs)
        gradients = cast(list[tf.Tensor | None], tape.gradient(loss, self.weights))
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
            batch, loss = -1, tf.constant(0.0)
            for batch, (inputs_batch, targets_batch) in enumerate(batches):
                loss = self._train_step(inputs_batch, targets_batch)
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
            NaiveDense(HIDDEN_LAYER_SIZE, activation="relu"),
            NaiveDense(N_CLASSES, activation="softmax"),
        ])

        self._model.compile(
            optimizer="rmsprop",
            # Losses:
            #   categorical_crossentropy:        Use one-hot labels
            #   sparse_categorical_crossentropy: Use integer labels (better performance and better memory efficiency)
            loss="sparse_categorical_crossentropy",
        )

    @property
    def weights(self) -> list[F32Array]:
        return [cast(tf.Tensor, w).numpy() for w in self._model.weights]

    # In this mini-batch SGD, 2345 (= 5 * ceil(60000 / 128)) weight updates are performed.
    def train(self) -> None:
        if self._model.loss_fn is keras.losses.categorical_crossentropy:
            train_labels = to_one_hot(self._train_labels, dimension=N_CLASSES)
        else:
            train_labels = self._train_labels

        self._model.fit(
            tf.convert_to_tensor(self._train_images),
            tf.convert_to_tensor(train_labels, dtype=tf.int32),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=True,
        )

    def evaluate(self) -> Evaluation:
        if self._model.loss_fn is keras.losses.categorical_crossentropy:
            test_labels = to_one_hot(self._test_labels, dimension=N_CLASSES)
        else:
            test_labels = self._test_labels

        inputs = tf.convert_to_tensor(self._test_images)
        targets = tf.convert_to_tensor(test_labels, dtype=tf.int32)

        probs = self._model(inputs)
        loss = self._model.compute_loss(targets, probs)

        preds = tf.argmax(probs, axis=1, output_type=tf.int32)
        if self._model.loss_fn is keras.losses.categorical_crossentropy:
            targets = tf.convert_to_tensor(self._test_labels, dtype=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(preds == targets, tf.float32))
        return Evaluation(float(loss.numpy()), float(accuracy.numpy()))

    def predict_probs(self, images: U8Array) -> F32Array:
        assert images.ndim == 3 and images.shape[1] == IMAGE_HEIGHT and images.shape[2] == IMAGE_WIDTH
        images_f32: F32Array = preprocess_data(images)
        probs: tf.Tensor = self._model(tf.convert_to_tensor(images_f32))
        assert tuple(probs.shape) == (images.shape[0], N_CLASSES)
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


def _run() -> None:
    set_random_seed_for("tensorflow", seed=None)
    tf_set_log_level(argv_index=1, default_level=1)

    nn = NeuralNetwork()
    train_time = elapsed_time(nn.train)
    evaluation = nn.evaluate()
    print(f"TRAIN TIME: {train_time:.2f}s, ACCURACY: {evaluation.accuracy:.2%}, LOSS: {evaluation.loss:.4f}")

    _, (test_images, test_labels) = load_raw_data()
    assert nn.predict_one(test_images[0]) == test_labels[0]


if __name__ == "__main__":
    _run()

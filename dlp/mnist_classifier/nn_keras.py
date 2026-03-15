#!../../.venv/bin/python
from __future__ import annotations

import keras
from keras import layers

# import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\ntest accuracy: {test_acc}")

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(len(predictions))  # 10
print(predictions[0])
print(predictions[0].argmax())
print(predictions[0][7])
print(test_labels[0])

#!../.venv/bin/python
from __future__ import annotations

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Physical devices  : {tf.config.list_physical_devices()}")

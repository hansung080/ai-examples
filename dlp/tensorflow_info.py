#!../.venv/bin/python
from __future__ import annotations

import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"Physical Devices:   {tf.config.list_physical_devices()}")

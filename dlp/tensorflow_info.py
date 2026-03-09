#!../.venv/bin/python3
from __future__ import annotations

import tensorflow as tf

print(f"tensorflow version: {tf.__version__}")
print(f"physical devices: {tf.config.list_physical_devices()}")

from enum import IntEnum
from typing import NamedTuple, Protocol

from common import F32Array, U8Array


class Evaluation(NamedTuple):
    accuracy: float


class Background(IntEnum):
    DARK = 0
    LIGHT = 1


class NeuralNetworkProtocol(Protocol):
    def train(self) -> None: ...
    def evaluate(self) -> Evaluation: ...
    def predict_proba(self, colors: U8Array | F32Array) -> F32Array: ...
    def predict(self, colors: U8Array | F32Array) -> U8Array: ...
    def predict_one(self, r: int, g: int, b: int) -> Background: ...

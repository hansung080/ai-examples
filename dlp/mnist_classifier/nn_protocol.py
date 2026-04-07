from enum import IntEnum
from typing import NamedTuple, Protocol

from common import F32Array, U8Array


class Evaluation(NamedTuple):
    loss: float
    accuracy: float


class Digit(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


class NeuralNetworkProtocol(Protocol):
    def train(self) -> None: ...
    def evaluate(self) -> Evaluation: ...
    def predict_probs(self, images: U8Array) -> F32Array: ...
    def predict(self, images: U8Array) -> U8Array: ...
    def predict_one(self, image: U8Array) -> Digit: ...

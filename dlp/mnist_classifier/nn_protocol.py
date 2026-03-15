from typing import Protocol


class NeuralNetworkProtocol(Protocol):
    def train(self) -> None: ...


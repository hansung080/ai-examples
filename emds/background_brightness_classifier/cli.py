#!../../.venv/bin/python3
from __future__ import annotations

import sys

from common import CLASSIFICATION_THRESHOLD, elapsed_time
from nn_from_scratch import NeuralNetwork as ScratchNeuralNetwork
from nn_protocol import NeuralNetworkProtocol
from nn_sklearn import NeuralNetwork as SklearnNeuralNetwork

CLASSIFICATION_THRESHOLD_FLOAT = float(CLASSIFICATION_THRESHOLD)


class FontColorRecommender:
    def __init__(self, nn: NeuralNetworkProtocol) -> None:
        self.nn = nn

    def recommend(self, r: int, g: int, b: int) -> str:
        pred = self.nn.predict(r, g, b)
        if pred >= CLASSIFICATION_THRESHOLD_FLOAT:
            return f"BLACK font recommended on light background ({r}, {g}, {b})"
        else:
            return f"WHITE font recommended on dark background ({r}, {g}, {b})"


def select_neural_network() -> tuple[NeuralNetworkProtocol, str]:
    print("[1] nn_from_scratch")
    print("[2] nn_sklearn")
    user_input = input("> Select a neural network: ").strip()
    match user_input:
        case "1" | "nn_from_scratch":
            return ScratchNeuralNetwork(), "nn_from_scratch"
        case "2" | "nn_sklearn":
            return SklearnNeuralNetwork(), "nn_sklearn"
        case _:
            print(f"error: unknown neural network: {user_input!r}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    nn, name = select_neural_network()
    train_time = elapsed_time(nn.train)
    print(f"{name}: train time: {train_time:.2f}s, accuracy: {nn.evaluate():.2%}\n")

    fcr = FontColorRecommender(nn)

    while True:
        user_input = input("background-color(r, g, b)> ").strip()
        if not user_input:
            continue
        elif user_input == "exit":
            break

        try:
            r, g, b = user_input.split(",")
            print(fcr.recommend(int(r), int(g), int(b)))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

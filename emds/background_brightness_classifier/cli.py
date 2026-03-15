#!../../.venv/bin/python
from __future__ import annotations

import sys

from common import elapsed_time
from nn_from_scratch import NeuralNetwork as ScratchNeuralNetwork
from nn_protocol import Background, NeuralNetworkProtocol
from nn_sklearn import NeuralNetwork as SklearnNeuralNetwork


class FontColorRecommender:
    def __init__(self, nn: NeuralNetworkProtocol) -> None:
        self.nn = nn

    def recommend(self, r: int, g: int, b: int) -> str:
        match self.nn.predict_one(r, g, b):
            case Background.DARK:
                return f"WHITE font recommended on dark background ({r}, {g}, {b})"
            case Background.LIGHT:
                return f"BLACK font recommended on light background ({r}, {g}, {b})"


def select_neural_network() -> tuple[NeuralNetworkProtocol, str]:
    print("[1] nn_from_scratch")
    print("[2] nn_sklearn")
    user_input = input("> Select a neural network: ").strip()
    match user_input:
        case "1" | "nn_from_scratch":
            return ScratchNeuralNetwork(seed=42), "nn_from_scratch"
        case "2" | "nn_sklearn":
            return SklearnNeuralNetwork(random_state=42), "nn_sklearn"
        case _:
            print(f"error: unknown neural network: {user_input!r}", file=sys.stderr)
            sys.exit(1)


def validate_rgb(r: str, g: str, b: str) -> tuple[int, int, int]:
    rgb = int(r), int(g), int(b)
    for c in rgb:
        if c < 0 or c > 255:
            raise ValueError(f"{c} is out of range [0, 255]")
    return rgb


def main() -> None:
    nn, name = select_neural_network()
    train_time = elapsed_time(nn.train)
    print(f"{name}: train time: {train_time:.2f}s, accuracy: {nn.evaluate().accuracy:.2%}\n")

    fcr = FontColorRecommender(nn)

    while True:
        user_input = input("background-color(r, g, b)> ").strip()
        if not user_input:
            continue
        elif user_input == "exit":
            break

        try:
            r, g, b = user_input.split(",")
            print(fcr.recommend(*validate_rgb(r, g, b)))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

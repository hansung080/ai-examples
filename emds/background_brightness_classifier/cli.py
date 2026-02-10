#!../../.venv/bin/python3
from __future__ import annotations

import sys

from common import CLASSIFICATION_THRESHOLD
from nn_from_scratch import NeuralNetwork


class FontColorRecommender:
    def __init__(self, nn: NeuralNetwork) -> None:
        self.nn = nn

    def recommend(self, r: int, g: int, b: int) -> str:
        pred = self.nn.predict(r, g, b)
        if pred >= CLASSIFICATION_THRESHOLD:
            return f"BLACK font recommended on light background ({r}, {g}, {b})"
        else:
            return f"WHITE font recommended on dark background ({r}, {g}, {b})"


def main() -> None:
    nn = NeuralNetwork()
    nn.train()

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

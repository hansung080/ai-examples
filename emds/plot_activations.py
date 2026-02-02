#!../.venv/bin/python3
from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import matplotlib as mpl
import sympy as sp
from matplotlib.backends import BackendFilter, backend_registry
from pathlib import Path

ALLOWED_ACTIVATIONS = ("relu", "sigmoid")


def print_usage() -> None:
    filename = Path(__file__).name
    print(f"usage: ./{filename} {{{' | '.join(ALLOWED_ACTIVATIONS)}}}", file=sys.stderr)


@dataclass(frozen=True)
class Args:
    activation: str

    @classmethod
    def from_argv(cls, argv: Sequence[str]) -> Self:
        args = argv[1:]
        if len(args) != 1:
            raise ValueError(f"invalid number of arguments: expected 1, got {len(args)}")

        activation = args[0]
        if activation not in ALLOWED_ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation!r}")
        return cls(activation)

    @classmethod
    def from_argv_or_exit(cls, argv: Sequence[str]) -> Self:
        try:
            return cls.from_argv(argv)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            print_usage()
            sys.exit(1)


def is_interactive_backend() -> bool:
    return mpl.get_backend() in backend_registry.list_builtin(BackendFilter.INTERACTIVE)


def ensure_interactive_backend() -> None:
    if not is_interactive_backend():
        print(f"error: matplotlib backend {mpl.get_backend()!r} is not an interactive backend", file=sys.stderr)
        sys.exit(1)


def plot_relu() -> None:
    x = sp.symbols("x", real=True)
    relu = sp.Max(sp.S.Zero, x)
    sp.plot(
        relu,
        (x, -5, 5),
        title="ReLU(x) = max(0, x)",
        xlabel="x",
        ylabel="ReLU(x)",
    )


def plot_sigmoid() -> None:
    x = sp.symbols("x", real=True)
    sigmoid = 1 / (1 + sp.exp(-x))
    sp.plot(
        sigmoid,
        (x, -5, 5),
        title="Sigmoid(x) = 1 / (1 + exp(-x))",
        xlabel="x",
        ylabel="Sigmoid(x)",
    )


def main() -> None:
    args = Args.from_argv_or_exit(sys.argv)
    ensure_interactive_backend()

    match args.activation:
        case "relu":
            plot_relu()
        case "sigmoid":
            plot_sigmoid()
        case _:
            raise AssertionError(f"unexpected activation: {args.activation!r}")


if __name__ == "__main__":
    main()

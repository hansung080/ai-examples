#!../../.venv/bin/python3
from __future__ import annotations

import sympy as sp
from sympy.core.expr import Expr

a, b = sp.symbols("a b")

assert 2 * (a + b) == 2 * a + 2 * b                       # cheap equivalence check (using canonical form)
assert (2 * (a + b)).equals(2 * a + 2 * b)                # logical equivalence check (best effort)
assert sp.simplify((2 * (a + b)) - (2 * a + 2 * b)) == 0  # robust equivalence check (best for unit test)


def relu(z: Expr) -> Expr:
    return sp.Max(sp.S.Zero, z)


def sigmoid(z: Expr) -> Expr:
    return 1 / (1 + sp.exp(-z))


x, y, w1, b1, z1, a1, w2, b2, z2, a2 = sp.symbols("x y w1 b1 z1 a1 w2 b2 z2 a2")

# dL/dA2
loss = (a2 - y) ** 2
dl_da2 = sp.diff(loss, a2)
assert dl_da2 == 2 * a2 - 2 * y

# dA2/dZ2
_a2 = sigmoid(z2)
da2_dz2 = sp.diff(_a2, z2)
assert da2_dz2 == sp.exp(-z2) / (1 + sp.exp(-z2)) ** 2

# dZ2/dW2
_z2 = a1 * w2 + b2
dz2_dw2 = sp.diff(_z2, w2)
assert dz2_dw2 == a1

# dZ2/dB2
dz2_db2 = sp.diff(_z2, b2)
assert dz2_db2 == 1

# dZ2/dA1
dz2_da1 = sp.diff(_z2, a1)
assert dz2_da1 == w2

# dA1/dZ1
_a1 = relu(z1)
da1_dz1 = sp.diff(_a1, z1)
assert da1_dz1 == sp.Heaviside(z1)  # 0 if z1 < 0, 0.5 if z1 == 0, or 1 if z1 > 0

# dZ1/dW1
_z1 = x * w1 + b1
dz1_dw1 = sp.diff(_z1, w1)
assert dz1_dw1 == x

# dZ1/dB1
dz1_db1 = sp.diff(_z1, b1)
assert dz1_db1 == 1

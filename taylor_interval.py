import numpy as np
import mpmath as mp

mp.iv.dps = 50
from typing import Any

import mpmath as mp
import numpy as np

mp.iv.dps = 50


def _iv_interval(lower: float, upper: float) -> Any:
    return mp.iv.mpf((lower, upper))  # type: ignore[arg-type]


class IntervalBox:
    """Hyperrectangle in parameter space."""

    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = np.array(radius, dtype=float)

    def intervals(self):
        return [_iv_interval(c - r, c + r) for c, r in zip(self.center, self.radius)]


def interval_dot(a, b):
    """Interval dot product."""
    out: Any = mp.iv.mpf(0)

    for x, y in zip(a, b):
        out += x * y

    return out


def third_order_remainder(M3, radius):
    """Compute rigorous Taylor remainder."""
    norm = np.linalg.norm(radius)
    return M3 / 6 * norm**3


def quadratic_taylor_bound(energy, gradient, hessian_interval, box):
    """Compute Taylor enclosure with quadratic terms only."""
    qc = box.center
    E0: Any = mp.iv.mpf(energy(qc))

    dq = [_iv_interval(-r, r) for r in box.radius]
    g = gradient(qc)
    linear = interval_dot(g, dq)

    H = hessian_interval(box.intervals())

    quadratic: Any = mp.iv.mpf(0)
    n = len(dq)

    for i in range(n):
        for j in range(n):
            quadratic += dq[i] * H[i][j] * dq[j]

    quadratic = quadratic * mp.iv.mpf("0.5")  # type: ignore[arg-type]
    return E0 + linear + quadratic


def quadratic_cubic_taylor_bound(
    energy,
    gradient,
    hessian_interval,
    third_derivative_bound,
    box,
):
    """Compute Taylor enclosure with quadratic and cubic terms."""
    qc = box.center
    E0: Any = mp.iv.mpf(energy(qc))

    dq = [_iv_interval(-r, r) for r in box.radius]
    g = gradient(qc)
    linear = interval_dot(g, dq)

    H = hessian_interval(box.intervals())

    quadratic: Any = mp.iv.mpf(0)
    n = len(dq)

    for i in range(n):
        for j in range(n):
            quadratic += dq[i] * H[i][j] * dq[j]

    quadratic = quadratic * 0.5

    M3 = third_derivative_bound(box.intervals())
    R3 = third_order_remainder(M3, box.radius)
    remainder = _iv_interval(-R3, R3)

    return E0 + linear + quadratic + remainder


def third_derivative_norm_bound(T):
    """Frobenius norm bound of an interval third derivative tensor."""
    total: Any = mp.iv.mpf(0)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            for k in range(T.shape[2]):
                total += T[i, j, k] ** 2

    return mp.sqrt(total)


def prune_cell(bound, best_energy):
    """Branch-and-bound pruning rule."""
    lower = float(bound.a)
    return lower > best_energy
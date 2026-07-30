import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from taylor_interval import (
    IntervalBox,
    prune_cell,
    quadratic_cubic_taylor_bound,
    quadratic_taylor_bound,
)
from derivative import thomson_gradient, thomson_hessian
from energy import thompson_energy


def _iv_interval(lower, upper):
    return mp.iv.mpf((lower, upper))  # type: ignore[arg-type]


def flat_to_angles(q):
    return [(q[i], q[i + 1]) for i in range(0, len(q), 2)]


def flat_to_cartesian(q):
    angles = flat_to_angles(q)
    return np.array([
        [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ]
        for theta, phi in angles
    ])


def exact_energy(q):
    return thompson_energy(flat_to_angles(q))


def exact_gradient(q):
    return thomson_gradient(flat_to_cartesian(q)).ravel()


def exact_hessian_interval(q):
    H = thomson_hessian(flat_to_cartesian(q))
    return [[_iv_interval(float(H[i, j]), float(H[i, j])) for j in range(H.shape[1])] for i in range(H.shape[0])]


def test_quadratic_taylor_bound_n2_contains_exact_energy():
    center = [0.0, 0.0, 0.0, math.pi]
    box = IntervalBox(center=center, radius=[0.0] * 4)

    bound = quadratic_taylor_bound(
        energy=exact_energy,
        gradient=exact_gradient,
        hessian_interval=exact_hessian_interval,
        box=box,
    )

    exact = mp.mpf(exact_energy(center))
    assert bound.a <= exact <= bound.b


def test_quadratic_cubic_taylor_bound_n3_contains_exact_energy():
    center = [
        0.0,
        0.0,
        0.0,
        math.pi / 2,
        2.0 * math.pi / 3.0,
        math.pi / 2,
    ]
    box = IntervalBox(center=center, radius=[0.0] * 6)

    bound = quadratic_cubic_taylor_bound(
        energy=exact_energy,
        gradient=exact_gradient,
        hessian_interval=exact_hessian_interval,
        third_derivative_bound=lambda intervals: 0.0,
        box=box,
    )

    exact = mp.mpf(exact_energy(center))
    assert bound.a <= exact <= bound.b


def test_prune_cell_uses_lower_bound():
    interval = _iv_interval(1.0, 2.0)
    assert prune_cell(interval, best_energy=0.9)
    assert not prune_cell(interval, best_energy=1.1)

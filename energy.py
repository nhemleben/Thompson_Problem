import numpy as np
import importlib
import builtins
from functools import lru_cache
from geometry import spherical_to_cart

def _load_njit():
    try:
        return importlib.import_module("numba").njit, True
    except ImportError:
        def _identity_njit(*args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
        return _identity_njit, False


njit, NUMBA_ENABLED = _load_njit()

if not getattr(builtins, "_THOMPSON_NUMBA_STARTUP_PRINTED", False):
    status = "enabled" if NUMBA_ENABLED else "disabled"
    print(f"[startup] Numba JIT: {status}")
    setattr(builtins, "_THOMPSON_NUMBA_STARTUP_PRINTED", True)


@njit(cache=True)
def _thompson_energy_from_xyz(xyz):
    n = xyz.shape[0]
    E = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = xyz[i, 0] - xyz[j, 0]
            dy = xyz[i, 1] - xyz[j, 1]
            dz = xyz[i, 2] - xyz[j, 2]
            d2 = dx * dx + dy * dy + dz * dz

            if d2 == 0.0:
                return np.inf

            E += 1.0 / np.sqrt(d2)

    return E


@lru_cache(maxsize=200000)
def _spherical_to_cart_cached(theta, phi):
    point = spherical_to_cart(theta, phi)
    return float(point[0]), float(point[1]), float(point[2])


def thompson_energy(points):

    """
    Coulomb energy on sphere
    """

    if not points:
        return 0.0

    xyz = np.asarray([
        _spherical_to_cart_cached(theta, phi)
        for theta, phi in points
    ], dtype=np.float64)
    return float(_thompson_energy_from_xyz(xyz))
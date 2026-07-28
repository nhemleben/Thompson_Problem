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
def _max_pair_distance(c1, c2):
    maxd = 0.0

    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dx = c1[i, 0] - c2[j, 0]
            dy = c1[i, 1] - c2[j, 1]
            dz = c1[i, 2] - c2[j, 2]
            d = np.sqrt(dx * dx + dy * dy + dz * dz)

            if d > maxd:
                maxd = d

    return maxd


def corner_points(box):

    theta_lo = box[0].lo
    theta_hi = box[0].hi
    phi_lo = box[1].lo
    phi_hi = box[1].hi

    return _corner_points_cached(theta_lo, theta_hi, phi_lo, phi_hi)


@lru_cache(maxsize=200000)
def _corner_points_cached(theta_lo, theta_hi, phi_lo, phi_hi):

    theta=[ theta_lo, theta_hi ]
    phi=[ phi_lo, phi_hi ]

    pts=[]

    for t in theta:

        for p in phi:

            pts.append(
                spherical_to_cart(t,p)
            )

    return np.asarray(pts, dtype=np.float64)



def pair_lower_bound(box1,box2):

    """ Minimum possible Coulomb contribution """

    c1=corner_points(box1)
    c2=corner_points(box2)

    maxd = _max_pair_distance(c1, c2)

    if maxd == 0:
        return float("inf")

    return 1/maxd



def energy_lower_bound(cell):

    E=0

    bounds =[pr.bounds for pr in cell.particle_ranges]

    n=len(bounds)

    for i in range(n):
        for j in range(i+1,n):

            E += pair_lower_bound( bounds[i], bounds[j])

    return E


def _child_lb_task(args):

    parent_lb, old_terms, split_particle_index, child_bounds = args
    n = len(child_bounds)
    split_child_bounds = child_bounds[split_particle_index]

    new_terms = 0.0
    for k in range(n):
        if k == split_particle_index:
            continue
        new_terms += pair_lower_bound(split_child_bounds, child_bounds[k])

    return parent_lb - old_terms + new_terms


def energy_lower_bound_children(
    parent_cell,
    parent_lb,
    children,
    split_particle_index,
    pool=None,
):

    if split_particle_index is None:
        return [energy_lower_bound(child) for child in children]

    parent_bounds = [pr.bounds for pr in parent_cell.particle_ranges]
    n = len(parent_bounds)

    old_terms = 0.0
    split_parent_bounds = parent_bounds[split_particle_index]

    for k in range(n):
        if k == split_particle_index:
            continue
        old_terms += pair_lower_bound(split_parent_bounds, parent_bounds[k])

    child_bounds_list = [[pr.bounds for pr in child.particle_ranges] for child in children]

    if pool is not None and len(child_bounds_list) > 1:
        tasks = [
            (parent_lb, old_terms, split_particle_index, child_bounds)
            for child_bounds in child_bounds_list
        ]
        return pool.map(_child_lb_task, tasks)

    return [
        _child_lb_task((parent_lb, old_terms, split_particle_index, child_bounds))
        for child_bounds in child_bounds_list
    ]
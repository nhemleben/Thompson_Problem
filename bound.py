import numpy as np
from geometry import spherical_to_cart
from known_optimal import L_n
import importlib
import builtins
from functools import lru_cache

def d_min(n):
    minimum_radius = 1/(L_n(n) - (n-2)/2)
    return minimum_radius

def angle_min(n):
    alpha_min = 2 * np.arcsin( d_min(n) /2)
    return alpha_min


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
    status = "active" if NUMBA_ENABLED else "disabled"
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


@njit(cache=True)
def _min_sep_pair_possible_kernel(
    c1,
    c2,
    use_dist,
    dist_threshold_sq,
    use_angle,
    cos_threshold,
):
    max_dist_sq = 0.0
    min_dot = 1e300

    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dx = c1[i, 0] - c2[j, 0]
            dy = c1[i, 1] - c2[j, 1]
            dz = c1[i, 2] - c2[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq

            if use_angle:
                dot = c1[i, 0] * c2[j, 0] + c1[i, 1] * c2[j, 1] + c1[i, 2] * c2[j, 2]
                if dot < min_dot:
                    min_dot = dot

    if use_dist and max_dist_sq < dist_threshold_sq:
        return False

    if use_angle and min_dot > cos_threshold:
        return False

    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dx = c1[i, 0] - c2[j, 0]
            dy = c1[i, 1] - c2[j, 1]
            dz = c1[i, 2] - c2[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if use_dist and dist_sq < dist_threshold_sq:
                continue

            if use_angle:
                dot = c1[i, 0] * c2[j, 0] + c1[i, 1] * c2[j, 1] + c1[i, 2] * c2[j, 2]
                if dot > cos_threshold:
                    continue

            return True

    return False


@lru_cache(maxsize=500000)
def _min_separation_pair_possible_cached(
    theta1_lo,
    theta1_hi,
    phi1_lo,
    phi1_hi,
    theta2_lo,
    theta2_hi,
    phi2_lo,
    phi2_hi,
    use_dist,
    dist_threshold_sq,
    use_angle,
    cos_threshold,
):
    c1 = _corner_points_cached(theta1_lo, theta1_hi, phi1_lo, phi1_hi)
    c2 = _corner_points_cached(theta2_lo, theta2_hi, phi2_lo, phi2_hi)

    return bool(
        _min_sep_pair_possible_kernel(
            c1,
            c2,
            use_dist,
            dist_threshold_sq,
            use_angle,
            cos_threshold,
        )
    )


def min_separation_pair_possible(
    box1,
    box2,
    d_min_sq=None,
    cos_alpha_min=None,
    epsilon=1e-15,
):
    use_dist = d_min_sq is not None
    use_angle = cos_alpha_min is not None

    if not use_dist and not use_angle:
        return True

    eps = float(epsilon)
    eps_sq = eps * eps
    dist_threshold_sq = 0.0

    if use_dist:
        dist_threshold_sq = max(0.0, float(d_min_sq) - eps_sq)

    cos_threshold = 0.0
    if use_angle:
        cos_threshold = float(cos_alpha_min) + eps

    return _min_separation_pair_possible_cached(
        float(box1[0].lo),
        float(box1[0].hi),
        float(box1[1].lo),
        float(box1[1].hi),
        float(box2[0].lo),
        float(box2[0].hi),
        float(box2[1].lo),
        float(box2[1].hi),
        bool(use_dist),
        float(dist_threshold_sq),
        bool(use_angle),
        float(cos_threshold),
    )

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


def build_child_lb_tasks(parent_cell, parent_lb, children, split_particle_index):

    if split_particle_index is None:
        return None

    parent_bounds = [pr.bounds for pr in parent_cell.particle_ranges]
    n = len(parent_bounds)

    old_terms = 0.0
    split_parent_bounds = parent_bounds[split_particle_index]

    for k in range(n):
        if k == split_particle_index:
            continue
        old_terms += pair_lower_bound(split_parent_bounds, parent_bounds[k])

    child_bounds_list = [[pr.bounds for pr in child.particle_ranges] for child in children]

    return [
        (parent_lb, old_terms, split_particle_index, child_bounds)
        for child_bounds in child_bounds_list
    ]


def evaluate_child_lb_tasks(tasks, pool=None):

    if not tasks:
        return []

    if pool is not None and len(tasks) > 1:
        return pool.map(_child_lb_task, tasks)

    return [_child_lb_task(task) for task in tasks]


def energy_lower_bound_children(
    parent_cell,
    parent_lb,
    children,
    split_particle_index,
    pool=None,
):

    if split_particle_index is None:
        return [energy_lower_bound(child) for child in children]

    tasks = build_child_lb_tasks(
        parent_cell,
        parent_lb,
        children,
        split_particle_index
    )

    return evaluate_child_lb_tasks(tasks, pool=pool)
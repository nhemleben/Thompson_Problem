# pyright: reportMissingModuleSource=false

from __future__ import annotations

import argparse
import heapq
import math
import multiprocessing as mp_pool
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import Any, Callable

import mpmath as mp
import numpy as np
import sympy as sp  # type: ignore[import-untyped]

from inital_part import initial_cell
from bound import d_min
from partition import split_with_index
from energy import thompson_energy
from search import (
    _build_even_mesh,
    _cell_charts,
    _center_config,
    _min_separation_state_from_cell,
    _min_separation_state_from_parent,
    _ordered_theta_center,
    _ordered_theta_possible,
    _respects_min_separation,
)
from visualizations import visualize_final_minimum, visualize_parameter_mesh
from visualizations.global_visualize import draw_global_search

mp.iv.dps = 50
sp_any: Any = sp
_TAYLOR_CACHE_VERSION = 1
_WORKER_MODEL: TaylorModel | None = None
_WORKER_N: int | None = None
_WORKER_IV_DPS: int | None = None
_WORKER_INITIAL_CELL_MODE: str | None = None
_REUSABLE_POOLS: dict[tuple[int, int, int, str], Any] = {}
_CASCADE_EXIT_LABELS = {
    1: "lipschitz_prune",
    2: "linear_prune",
    3: "quadratic_prune",
    4: "cubic_exit",
}


def _set_iv_dps(iv_dps: int) -> int:
    value = int(iv_dps)
    if value < 5:
        raise ValueError("iv_dps must be >= 5")
    mp.iv.dps = value
    return value


def _taylor_cache_path(n: int) -> Path:
    cache_dir = Path(__file__).resolve().parent / "symbolic_derivatives"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"taylor_model_n{n}_v{_TAYLOR_CACHE_VERSION}.pkl"


def _load_symbolic_derivatives(n: int):
    cache_path = _taylor_cache_path(n)

    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    if payload.get("version") != _TAYLOR_CACHE_VERSION:
        return None

    if payload.get("n") != n:
        return None

    return payload


def _save_symbolic_derivatives(n: int, payload: dict[str, Any]) -> None:
    cache_path = _taylor_cache_path(n)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temp_path.replace(cache_path)


def _build_symbolic_payload(n: int) -> dict[str, Any]:
    theta, phi, variables = _particle_symbols(n)
    energy_expr = _spherical_energy_expression(theta, phi)
    gradient_expr = [sp_any.diff(energy_expr, variable) for variable in variables]
    hessian_expr = sp_any.Matrix(gradient_expr).jacobian(variables)
    hessian_expr_list = hessian_expr.tolist()
    third_expr = [
        [
            [sp_any.diff(energy_expr, vi, vj, vk) for vk in variables]
            for vj in variables
        ]
        for vi in variables
    ]

    return {
        "version": _TAYLOR_CACHE_VERSION,
        "n": n,
        "energy_expr": energy_expr,
        "gradient_expr": gradient_expr,
        "hessian_expr_list": hessian_expr_list,
        "third_expr": third_expr,
    }


def materialize_symbolic_derivatives(n: int, force: bool = False) -> Path:
    cache_path = _taylor_cache_path(n)

    if force or _load_symbolic_derivatives(n) is None:
        payload = _build_symbolic_payload(n)
        _save_symbolic_derivatives(n, payload)

    return cache_path


def _iv_interval(lower: float, upper: float) -> Any:
    return mp.iv.mpf((lower, upper))  # type: ignore[arg-type]


@dataclass
class TaylorModel:
    energy: Callable[[list[float] | np.ndarray], Any]
    gradient: Callable[[list[float] | np.ndarray], np.ndarray]
    hessian: Callable[[list[float] | np.ndarray], np.ndarray]
    third_derivative: Callable[[list[float] | np.ndarray], np.ndarray]
    third_derivative_usable: bool = True


def _particle_symbols(n: int):
    theta = sp_any.symbols(f"theta0:{n}", real=True)
    phi = sp_any.symbols(f"phi0:{n}", real=True)
    variables: list[sp.Symbol] = []

    for i in range(n):
        variables.append(theta[i])
        variables.append(phi[i])

    return theta, phi, variables


def _spherical_energy_expression(theta, phi):
    energy = 0

    for i in range(len(theta)):
        for j in range(i + 1, len(theta)):
            cosine = (
                sp_any.cos(phi[i]) * sp_any.cos(phi[j])
                + sp_any.sin(phi[i]) * sp_any.sin(phi[j]) * sp_any.cos(theta[i] - theta[j])
            )
            energy += 1 / sp_any.sqrt(2 - 2 * cosine)

    return energy


@lru_cache(maxsize=None)
def build_taylor_model(n: int) -> TaylorModel:
    theta, phi, variables = _particle_symbols(n)

    cached = _load_symbolic_derivatives(n)
    if cached is None:
        materialize_symbolic_derivatives(n)
        cached = _load_symbolic_derivatives(n)

    if cached is None:
        raise RuntimeError(f"Failed to load symbolic derivatives for n={n}")

    energy_expr = cached["energy_expr"]
    gradient_expr = cached["gradient_expr"]
    hessian_expr_list = cached["hessian_expr_list"]
    third_expr = cached["third_expr"]

    interval_modules = [{"sin": mp.iv.sin, "cos": mp.iv.cos, "sqrt": mp.iv.sqrt}, "mpmath"]

    energy_func = sp_any.lambdify(variables, energy_expr, modules=interval_modules)
    gradient_func = sp_any.lambdify(variables, gradient_expr, modules=interval_modules)
    hessian_func = sp_any.lambdify(variables, hessian_expr_list, modules=interval_modules)
    third_func = sp_any.lambdify(variables, third_expr, modules=interval_modules)

    def energy(values):
        return energy_func(*list(values))

    def gradient(values):
        return np.asarray(gradient_func(*list(values)), dtype=object).reshape(-1)

    def hessian(values):
        return np.asarray(hessian_func(*list(values)), dtype=object)

    def third_derivative(values):
        return np.asarray(third_func(*list(values)), dtype=object)

    return TaylorModel(
        energy=energy,
        gradient=gradient,
        hessian=hessian,
        third_derivative=third_derivative,
        third_derivative_usable=True,
    )


def _normalized_chart(chart: str) -> str:
    return str(chart).strip().lower()


def _canonicalize_flat_values(values, charts: list[str]) -> list[Any]:
    canonical = list(values)

    for particle_index, chart in enumerate(charts):
        if _normalized_chart(chart) != "antipodal_psi":
            continue

        phi_index = 2 * particle_index + 1
        canonical[phi_index] = math.pi - canonical[phi_index]

    return canonical


def _chart_sign_vector(charts: list[str]) -> np.ndarray:
    signs = np.ones(2 * len(charts), dtype=float)

    for particle_index, chart in enumerate(charts):
        if _normalized_chart(chart) == "antipodal_psi":
            signs[2 * particle_index + 1] = -1.0

    return signs


def _transform_gradient_from_canonical(gradient, sign_vector: np.ndarray):
    gradient_array = np.asarray(gradient, dtype=object).reshape(-1)
    transformed = np.asarray(gradient_array, dtype=object).copy()

    for index, sign in enumerate(sign_vector):
        if sign < 0:
            transformed[index] = -transformed[index]

    return transformed


def _transform_hessian_from_canonical(hessian, sign_vector: np.ndarray):
    hessian_array = np.asarray(hessian, dtype=object)
    transformed = np.asarray(hessian_array, dtype=object).copy()

    for i in range(transformed.shape[0]):
        for j in range(transformed.shape[1]):
            if sign_vector[i] * sign_vector[j] < 0:
                transformed[i, j] = -transformed[i, j]

    return transformed


def _transform_third_derivative_from_canonical(third, sign_vector: np.ndarray):
    third_array = np.asarray(third, dtype=object)
    transformed = np.asarray(third_array, dtype=object).copy()

    for i in range(transformed.shape[0]):
        for j in range(transformed.shape[1]):
            for k in range(transformed.shape[2]):
                if sign_vector[i] * sign_vector[j] * sign_vector[k] < 0:
                    transformed[i, j, k] = -transformed[i, j, k]

    return transformed


def _wrap_model_for_charts(model: TaylorModel, charts: list[str]) -> TaylorModel:
    if all(_normalized_chart(chart) == "standard" for chart in charts):
        return model

    sign_vector = _chart_sign_vector(charts)

    def energy(values):
        canonical_values = _canonicalize_flat_values(values, charts)
        return model.energy(canonical_values)

    def gradient(values):
        canonical_values = _canonicalize_flat_values(values, charts)
        canonical_gradient = model.gradient(canonical_values)
        return _transform_gradient_from_canonical(canonical_gradient, sign_vector)

    def hessian(values):
        canonical_values = _canonicalize_flat_values(values, charts)
        canonical_hessian = model.hessian(canonical_values)
        return _transform_hessian_from_canonical(canonical_hessian, sign_vector)

    def third_derivative(values):
        canonical_values = _canonicalize_flat_values(values, charts)
        canonical_third = model.third_derivative(canonical_values)
        return _transform_third_derivative_from_canonical(canonical_third, sign_vector)

    return TaylorModel(
        energy=energy,
        gradient=gradient,
        hessian=hessian,
        third_derivative=third_derivative,
        third_derivative_usable=model.third_derivative_usable,
    )


def _flat_center_and_radius(cell):
    center: list[float] = []
    radius: list[float] = []
    intervals: list[Any] = []

    for particle_range in cell.particle_ranges:
        for bound in particle_range.bounds:
            c = 0.5 * (bound.lo + bound.hi)
            r = 0.5 * (bound.hi - bound.lo)
            center.append(c)
            radius.append(r)
            intervals.append(_iv_interval(bound.lo, bound.hi))

    return np.asarray(center, dtype=float), np.asarray(radius, dtype=float), intervals


def _max_sin_sq(lo: float, hi: float) -> float:
    best = max(math.sin(lo) ** 2, math.sin(hi) ** 2)
    first_critical = math.ceil((lo - math.pi / 2) / math.pi)
    last_critical = math.floor((hi - math.pi / 2) / math.pi)

    if first_critical <= last_critical:
        return 1.0

    return best


def _spherical_metric_radius(cell) -> float:
    total = 0.0

    for particle_range in cell.particle_ranges:
        theta_bounds, phi_bounds = particle_range.bounds
        d_theta = 0.5 * (theta_bounds.hi - theta_bounds.lo)
        d_phi = 0.5 * (phi_bounds.hi - phi_bounds.lo)

        total += d_theta * d_theta
        total += _max_sin_sq(theta_bounds.lo, theta_bounds.hi) * d_phi * d_phi

    return math.sqrt(total)


def _cell_parameter_volume(cell) -> float:
    volume = 1.0
    free_dims = 0

    for particle_range in cell.particle_ranges:
        for dim, bounds in enumerate(particle_range.bounds):
            if particle_range.fixed[dim]:
                continue

            width = max(0.0, float(bounds.hi - bounds.lo))
            volume *= width
            free_dims += 1

    if free_dims == 0:
        return 1.0

    return volume


def _init_termination_tracker() -> dict[str, dict[int, float] | dict[int, int]]:
    return {
        "counts": {1: 0, 2: 0, 3: 0, 4: 0},
        "volumes": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    }


def _record_termination_exit(tracker, stage: int, cell) -> None:
    if tracker is None:
        return

    counts = tracker["counts"]
    volumes = tracker["volumes"]
    counts[stage] = counts[stage] + 1
    volumes[stage] = volumes[stage] + _cell_parameter_volume(cell)


def _record_termination_observation(tracker, stage: int | None, volume: float) -> None:
    if tracker is None or stage is None:
        return

    counts = tracker["counts"]
    volumes = tracker["volumes"]
    counts[stage] = counts[stage] + 1
    volumes[stage] = volumes[stage] + float(volume)


def _tensor_frobenius_norm_bound(tensor) -> Any:
    array = np.asarray(tensor, dtype=object)
    total = mp.iv.mpf(0)

    for index in np.ndindex(array.shape):
        total += array[index] ** 2

    lower = max(0.0, float(total.a))
    upper = float(total.b)
    return mp.iv.sqrt(_iv_interval(lower, upper))


def _interval_cartesian_point(theta_bounds, phi_bounds, chart="standard"):
    theta = _iv_interval(theta_bounds.lo, theta_bounds.hi)
    phi = _iv_interval(phi_bounds.lo, phi_bounds.hi)

    if _normalized_chart(chart) == "antipodal_psi":
        psi = phi
        return (
            mp.iv.sin(psi) * mp.iv.cos(theta),  # type: ignore[operator]
            mp.iv.sin(psi) * mp.iv.sin(theta),  # type: ignore[operator]
            -mp.iv.cos(psi),
        )

    return (
        mp.iv.sin(phi) * mp.iv.cos(theta),  # type: ignore[operator]
        mp.iv.sin(phi) * mp.iv.sin(theta),  # type: ignore[operator]
        mp.iv.cos(phi),
    )


def _pair_distance_lower_bound(bounds_a, bounds_b, chart_a="standard", chart_b="standard") -> float:
    x1, y1, z1 = _interval_cartesian_point(*bounds_a, chart=chart_a)
    x2, y2, z2 = _interval_cartesian_point(*bounds_b, chart=chart_b)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2  # type: ignore[operator]

    distance_sq = dx * dx + dy * dy + dz * dz
    return math.sqrt(max(0.0, float(distance_sq.a)))


def _fallback_third_derivative_bound(cell) -> float:
    pairwise_bound = 0.0
    bounds = [particle_range.bounds for particle_range in cell.particle_ranges]
    charts = [getattr(particle_range, "chart", "standard") for particle_range in cell.particle_ranges]

    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            d_min = _pair_distance_lower_bound(
                bounds[i],
                bounds[j],
                chart_a=charts[i],
                chart_b=charts[j],
            )
            if d_min <= 0.0:
                return float("inf")

            pairwise_bound += 256.0 / (d_min ** 4)

    return pairwise_bound


def _taylor_enclosure(cell, model: TaylorModel):
    center, radius, intervals = _flat_center_and_radius(cell)
    config = _center_config(cell)
    charts = _cell_charts(cell)

    center_energy = thompson_energy(config, charts=charts)
    qc_energy: Any = _iv_interval(center_energy, center_energy)
    gradient = model.gradient(center)
    hessian = model.hessian(intervals)

    dq = [_iv_interval(-r, r) for r in radius]

    linear: Any = mp.iv.mpf(0)
    for gi, dqi in zip(gradient, dq):
        linear += gi * dqi

    quadratic: Any = mp.iv.mpf(0)
    for i in range(len(dq)):
        for j in range(len(dq)):
            quadratic += dq[i] * hessian[i, j] * dq[j]

    quadratic = quadratic * _iv_interval(0.5, 0.5)

    if model.third_derivative_usable:
        try:
            third_derivative = model.third_derivative(intervals)
            third_norm = _tensor_frobenius_norm_bound(third_derivative)
            third_bound = float(third_norm.b)
        except Exception:
            model.third_derivative_usable = False
            third_bound = _fallback_third_derivative_bound(cell)
    else:
        third_bound = _fallback_third_derivative_bound(cell)

    remainder_radius = third_bound / 6.0 * _spherical_metric_radius(cell) ** 3
    remainder: Any = _iv_interval(-remainder_radius, remainder_radius)

    result: Any = qc_energy + linear + quadratic + remainder
    return result


def _cascading_taylor_lower_bound(
    cell,
    model: TaylorModel,
    best_energy: float | None = None,
    termination_tracker=None,
):
    center, radius, intervals = _flat_center_and_radius(cell)
    config = _center_config(cell)
    charts = _cell_charts(cell)
    center_energy = thompson_energy(config, charts=charts)
    metric_radius = _spherical_metric_radius(cell)

    # 1) Cheap: Lipschitz bound from center gradient norm
    gradient = model.gradient(center)

    def _abs_upper(value: Any) -> float:
        try:
            lo = float(value.a)
            hi = float(value.b)
            return max(abs(lo), abs(hi))
        except Exception:
            return abs(float(value))

    grad_l1 = sum(_abs_upper(gi) for gi in gradient)
    lipschitz_lb = center_energy - grad_l1 * metric_radius

    if best_energy is not None and lipschitz_lb >= best_energy:
        _record_termination_exit(termination_tracker, 1, cell)
        return None, center_energy, 1

    dq = [_iv_interval(-r, r) for r in radius]
    e0: Any = _iv_interval(center_energy, center_energy)

    # 2) Slightly more expensive: linear Taylor bound
    linear: Any = mp.iv.mpf(0)
    for gi, dqi in zip(gradient, dq):
        linear += gi * dqi

    linear_interval: Any = e0 + linear
    linear_lb = float(linear_interval.a)

    if best_energy is not None and linear_lb >= best_energy:
        _record_termination_exit(termination_tracker, 2, cell)
        return None, center_energy, 2

    # 3) Expensive: quadratic Taylor bound
    hessian = model.hessian(intervals)
    quadratic: Any = mp.iv.mpf(0)
    for i in range(len(dq)):
        for j in range(len(dq)):
            quadratic += dq[i] * hessian[i, j] * dq[j]

    quadratic = quadratic * _iv_interval(0.5, 0.5)
    quadratic_interval: Any = linear_interval + quadratic
    quadratic_lb = float(quadratic_interval.a)

    if best_energy is not None and quadratic_lb >= best_energy:
        _record_termination_exit(termination_tracker, 3, cell)
        return None, center_energy, 3

    # 4) Very expensive: cubic remainder bound
    if model.third_derivative_usable:
        try:
            third_derivative = model.third_derivative(intervals)
            third_norm = _tensor_frobenius_norm_bound(third_derivative)
            third_bound = float(third_norm.b)
        except Exception:
            model.third_derivative_usable = False
            third_bound = _fallback_third_derivative_bound(cell)
    else:
        third_bound = _fallback_third_derivative_bound(cell)

    remainder_radius = third_bound / 6.0 * metric_radius ** 3
    remainder: Any = _iv_interval(-remainder_radius, remainder_radius)

    cubic_interval: Any = quadratic_interval + remainder
    _record_termination_exit(termination_tracker, 4, cell)
    return float(cubic_interval.a), center_energy, 4


def _taylor_lower_bound(cell, model: TaylorModel) -> float:
    lower_bound, _, _ = _cascading_taylor_lower_bound(cell, model, best_energy=None)
    if lower_bound is None:
        return float("inf")
    return lower_bound


def _init_taylor_worker(n: int, iv_dps: int, initial_cell_mode: str):
    global _WORKER_MODEL, _WORKER_N, _WORKER_IV_DPS, _WORKER_INITIAL_CELL_MODE

    _WORKER_N = int(n)
    _WORKER_IV_DPS = _set_iv_dps(int(iv_dps))
    _WORKER_INITIAL_CELL_MODE = str(initial_cell_mode)
    base_model = build_taylor_model(_WORKER_N)
    root = initial_cell(_WORKER_N, mode=_WORKER_INITIAL_CELL_MODE)
    _WORKER_MODEL = _wrap_model_for_charts(base_model, _cell_charts(root))


def _get_reusable_pool(n: int, iv_dps: int, workers: int, initial_cell_mode: str):
    mode = str(initial_cell_mode)
    key = (int(n), int(iv_dps), int(workers), mode)
    pool = _REUSABLE_POOLS.get(key)

    if pool is not None:
        return pool

    pool = mp_pool.Pool(
        processes=int(workers),
        initializer=_init_taylor_worker,
        initargs=(int(n), int(iv_dps), mode),
    )
    _REUSABLE_POOLS[key] = pool
    return pool


def shutdown_reusable_worker_pools():
    for pool in _REUSABLE_POOLS.values():
        pool.close()
        pool.join()

    _REUSABLE_POOLS.clear()


def _child_taylor_lb_task(args):
    if len(args) == 2:
        child, best_energy = args
        model = _WORKER_MODEL

        if model is None:
            raise RuntimeError("Worker model not initialized; pool initializer was not run")

        lower_bound, _, stage = _cascading_taylor_lower_bound(
            child,
            model,
            best_energy=best_energy,
        )
        return lower_bound, stage, _cell_parameter_volume(child)

    n, child, iv_dps, best_energy, initial_cell_mode = args
    _set_iv_dps(iv_dps)
    base_model = build_taylor_model(n)
    root = initial_cell(int(n), mode=str(initial_cell_mode))
    model = _wrap_model_for_charts(base_model, _cell_charts(root))
    lower_bound, _, stage = _cascading_taylor_lower_bound(child, model, best_energy=best_energy)
    return lower_bound, stage, _cell_parameter_volume(child)


def _evaluate_child_lb_tasks(tasks, pool=None):
    if not tasks:
        return []

    if pool is not None and len(tasks) > 1:
        return pool.map(_child_taylor_lb_task, tasks)

    return [_child_taylor_lb_task(task) for task in tasks]


def search(
    n,
    target_depth=12,
    visualize_search=False,
    visualize_all_particles=False,
    visualize_mesh=False,
    show_progress=True,
    progress_update_every=10000,
    parallel_child_bounds=False,
    parallel_workers=mp_pool.cpu_count(),
    parallel_batch_size=32,
    iv_dps=50,
    visualize_final=True,
    d_min=None,
    alpha_min=None,
    use_min_separation=False,
    initial_mesh_side_length=0.1,
    initial_cell_mode="non-antipodal",
    reuse_worker_pool=True,
    measure_termination_volumes=False,
):
    iv_dps = _set_iv_dps(iv_dps)
    base_model = build_taylor_model(n)
    root = initial_cell(n, mode=initial_cell_mode)
    charts = _cell_charts(root)
    model = _wrap_model_for_charts(base_model, charts)
    tie_breaker = count()

    queue = []
    active_cells = []
    bounds = []

    best = float("inf")
    best_config = None

    use_min_separation = use_min_separation or (d_min is not None) or (alpha_min is not None)
    if use_min_separation and d_min is None:
        d_min = d_min(n)
    min_sep_cos_alpha = float(np.cos(alpha_min)) if alpha_min is not None else None
    min_sep_d_sq = float(d_min) * float(d_min) if d_min is not None else None
    termination_tracker = _init_termination_tracker() if measure_termination_volumes else None

    processed_nodes = 0
    estimated_total_nodes = ((2 ** (target_depth + 1)) - 1) / math.factorial(n - 2)
    progress_line_width = 0
    start_time = time.perf_counter()

    def _print_progress_line(current_nodes):
        nonlocal progress_line_width
        percent = min(100.0, (current_nodes / estimated_total_nodes) * 100)
        elapsed = time.perf_counter() - start_time
        if current_nodes > 0:
            seconds_per_1000 = (elapsed / current_nodes) * 1000
            rate_text = f"{seconds_per_1000:.3f}s/1000 nodes"
        else:
            rate_text = "n/a s/1000 nodes"
        line = (
            f"Progress: {percent:.1f}% "
            f"({current_nodes}/{estimated_total_nodes} estimated nodes, {rate_text})"
        )
        progress_line_width = max(progress_line_width, len(line))
        print(line.ljust(progress_line_width), end="\r", flush=True)

    if show_progress:
        _print_progress_line(0)

    pool = None
    pool_owned = False
    if parallel_child_bounds:
        workers = int(parallel_workers) if parallel_workers is not None else mp_pool.cpu_count()

        if reuse_worker_pool:
            pool = _get_reusable_pool(n, int(iv_dps), workers, str(initial_cell_mode))
            pool_owned = False
        else:
            pool = mp_pool.Pool(
                processes=workers,
                initializer=_init_taylor_worker,
                initargs=(int(n), int(iv_dps), str(initial_cell_mode)),
            )
            pool_owned = True

    try:
        initial_cells = _build_even_mesh(
            root,
            side_length=initial_mesh_side_length,
            cell_filter=_ordered_theta_possible,
        )

        if parallel_child_bounds and len(initial_cells) > 1:
            initial_tasks = [
                (cell, float("inf"))
                for cell in initial_cells
            ]
            initial_results = _evaluate_child_lb_tasks(initial_tasks, pool=pool)
            initial_lbs = []

            for lb_value, stage, volume in initial_results:
                _record_termination_observation(
                    termination_tracker,
                    stage,
                    volume,
                )
                initial_lbs.append(lb_value)
        else:
            initial_lbs = []
            for cell in initial_cells:
                lb_value, _, _ = _cascading_taylor_lower_bound(
                    cell,
                    model,
                    best_energy=None,
                    termination_tracker=termination_tracker,
                )
                if lb_value is None:
                    initial_lbs.append(float("inf"))
                else:
                    initial_lbs.append(lb_value)

        for cell, root_lb in zip(initial_cells, initial_lbs):
            if root_lb is not None:
                min_sep_state = None
                if use_min_separation:
                    min_sep_ok, min_sep_state = _min_separation_state_from_cell(
                        cell,
                        d_min=d_min,
                        alpha_min=alpha_min,
                        cos_alpha_min=min_sep_cos_alpha,
                        d_min_sq=min_sep_d_sq,
                    )
                    if not min_sep_ok:
                        continue

                heapq.heappush(queue, (root_lb, next(tie_breaker), cell, min_sep_state))

        use_batched_parallel = parallel_child_bounds and parallel_batch_size > 1

        while queue:
            if use_batched_parallel:
                batch_count = min(parallel_batch_size, len(queue))
                frontier = [heapq.heappop(queue) for _ in range(batch_count)]

                pending_children = []
                pending_children_states = []
                pending_tasks = []

                for lb, _, cell, parent_min_sep_state in frontier:
                    processed_nodes += 1

                    if show_progress and (processed_nodes % progress_update_every == 0):
                        _print_progress_line(processed_nodes)

                    if visualize_search:
                        active_cells.append(cell)
                        bounds.append(lb)

                    if lb >= best:
                        continue

                    if not _ordered_theta_possible(cell):
                        continue

                    if use_min_separation and parent_min_sep_state is None:
                        min_sep_ok, parent_min_sep_state = _min_separation_state_from_cell(
                            cell,
                            d_min=d_min,
                            alpha_min=alpha_min,
                            cos_alpha_min=min_sep_cos_alpha,
                            d_min_sq=min_sep_d_sq,
                        )
                        if not min_sep_ok:
                            continue

                    config = _center_config(cell)

                    refreshed_lb, center_energy, _ = _cascading_taylor_lower_bound(
                        cell,
                        model,
                        best_energy=best,
                        termination_tracker=termination_tracker,
                    )

                    # If any cascade stage proves lb >= best, this cell is
                    # rigorously pruned and must not be split.
                    if refreshed_lb is None:
                        continue

                    center_feasible = True
                    if use_min_separation:
                        center_feasible = _respects_min_separation(
                            config,
                            d_min=d_min,
                            alpha_min=alpha_min,
                            charts=charts,
                        )

                    if center_feasible:
                        exact_energy = center_energy
                    else:
                        exact_energy = float("inf")

                    if exact_energy < best:
                        if not _ordered_theta_center(config):
                            continue

                        best = exact_energy
                        best_config = config

                        if show_progress:
                            print()
                        print("new", best)

                    if cell.depth < target_depth:
                        children, split_particle_index = split_with_index(cell)
                        children = [
                            child for child in children if _ordered_theta_possible(child)
                        ]

                        if not children:
                            continue

                        feasible_children = []
                        feasible_states = []

                        for child in children:
                            if use_min_separation:
                                child_ok, child_state = _min_separation_state_from_parent(
                                    parent_min_sep_state,
                                    cell,
                                    child,
                                    split_particle_index,
                                    d_min=d_min,
                                    alpha_min=alpha_min,
                                    cos_alpha_min=min_sep_cos_alpha,
                                    d_min_sq=min_sep_d_sq,
                                )
                                if not child_ok:
                                    continue
                            else:
                                child_state = None

                            feasible_children.append(child)
                            feasible_states.append(child_state)

                        if not feasible_children:
                            continue

                        pending_children.extend(feasible_children)
                        pending_children_states.extend(feasible_states)

                        task_items = []
                        if pool is not None:
                            task_items = [(child, best) for child in feasible_children]
                        else:
                            task_items = [
                                (n, child, int(iv_dps), best, str(initial_cell_mode))
                                for child in feasible_children
                            ]

                        pending_tasks.extend(
                            task_items
                        )

                if pending_tasks:
                    child_results = _evaluate_child_lb_tasks(pending_tasks, pool=pool)

                    child_lbs = []
                    for child_lb, child_stage, child_volume in child_results:
                        _record_termination_observation(
                            termination_tracker,
                            child_stage,
                            child_volume,
                        )
                        child_lbs.append(child_lb)

                    for child, child_lb, child_state in zip(
                        pending_children,
                        child_lbs,
                        pending_children_states,
                    ):
                        if child_lb is not None and child_lb < best:
                            heapq.heappush(queue, (child_lb, next(tie_breaker), child, child_state))
            else:
                lb, _, cell, parent_min_sep_state = heapq.heappop(queue)
                processed_nodes += 1

                if show_progress and (
                    processed_nodes % progress_update_every == 0 or not queue
                ):
                    _print_progress_line(processed_nodes)

                if visualize_search:
                    active_cells.append(cell)
                    bounds.append(lb)

                if lb >= best:
                    continue

                if not _ordered_theta_possible(cell):
                    continue

                if use_min_separation and parent_min_sep_state is None:
                    min_sep_ok, parent_min_sep_state = _min_separation_state_from_cell(
                        cell,
                        d_min=d_min,
                        alpha_min=alpha_min,
                        cos_alpha_min=min_sep_cos_alpha,
                        d_min_sq=min_sep_d_sq,
                    )
                    if not min_sep_ok:
                        continue

                config = _center_config(cell)

                refreshed_lb, center_energy, _ = _cascading_taylor_lower_bound(
                    cell,
                    model,
                    best_energy=best,
                    termination_tracker=termination_tracker,
                )

                # If any cascade stage proves lb >= best, this cell is
                # rigorously pruned and must not be split.
                if refreshed_lb is None:
                    continue

                center_feasible = True
                if use_min_separation:
                    center_feasible = _respects_min_separation(
                        config,
                        d_min=d_min,
                        alpha_min=alpha_min,
                        charts=charts,
                    )

                if center_feasible:
                    exact_energy = center_energy
                else:
                    exact_energy = float("inf")

                if exact_energy < best:
                    if not _ordered_theta_center(config):
                        continue

                    best = exact_energy
                    best_config = config

                    if show_progress:
                        print()
                    print("new", best)

                if cell.depth < target_depth:
                    children, split_particle_index = split_with_index(cell)
                    children = [
                        child for child in children if _ordered_theta_possible(child)
                    ]

                    if not children:
                        continue

                    feasible_children = []
                    feasible_states = []

                    for child in children:
                        if use_min_separation:
                            child_ok, child_state = _min_separation_state_from_parent(
                                parent_min_sep_state,
                                cell,
                                child,
                                split_particle_index,
                                d_min=d_min,
                                alpha_min=alpha_min,
                                cos_alpha_min=min_sep_cos_alpha,
                                d_min_sq=min_sep_d_sq,
                            )
                            if not child_ok:
                                continue
                        else:
                            child_state = None

                        feasible_children.append(child)
                        feasible_states.append(child_state)

                    if not feasible_children:
                        continue

                    if pool is not None:
                        child_tasks = [(child, best) for child in feasible_children]
                    else:
                        child_tasks = [
                            (n, child, int(iv_dps), best, str(initial_cell_mode))
                            for child in feasible_children
                        ]
                    child_results = _evaluate_child_lb_tasks(child_tasks, pool=pool)
                    child_lbs = []

                    for child_lb, child_stage, child_volume in child_results:
                        _record_termination_observation(
                            termination_tracker,
                            child_stage,
                            child_volume,
                        )
                        child_lbs.append(child_lb)

                    for child, child_lb, child_state in zip(feasible_children, child_lbs, feasible_states):
                        if child_lb is not None and child_lb < best:
                            heapq.heappush(queue, (child_lb, next(tie_breaker), child, child_state))

        if show_progress and not queue:
            _print_progress_line(processed_nodes)
    finally:
        if pool is not None and pool_owned:
            pool.close()
            pool.join()

    elapsed_total = time.perf_counter() - start_time
    print("Total elapsed time:", elapsed_total)
    if processed_nodes > 0:
        average_seconds_per_node = elapsed_total / processed_nodes
        print(
            f"Average time: {average_seconds_per_node:.6f}s/node over {processed_nodes} nodes"
        )
    else:
        print("Average time: n/a s/node over 0 nodes")

    if measure_termination_volumes and termination_tracker is not None:
        counts = termination_tracker["counts"]
        volumes = termination_tracker["volumes"]
        total_count = sum(counts.values())
        total_volume = sum(volumes.values())

        print("Termination Exit Tracker (_cascading_taylor_lower_bound)")
        print(
            f"  total_evaluated_cells={total_count}, "
            f"total_evaluated_volume={total_volume:.12e}"
        )

        for stage in (1, 2, 3, 4):
            stage_count = counts[stage]
            stage_volume = volumes[stage]

            if total_count > 0:
                count_fraction = stage_count / total_count
            else:
                count_fraction = 0.0

            if total_volume > 0.0:
                volume_fraction = stage_volume / total_volume
            else:
                volume_fraction = 0.0

            print(
                f"  {_CASCADE_EXIT_LABELS[stage]:>16}: "
                f"count={stage_count} ({count_fraction:.2%}), "
                f"volume={stage_volume:.12e} ({volume_fraction:.2%})"
            )

    if visualize_final and best_config is not None:
        visualize_final_minimum.plot_final_minimum(best_config, best)

    if visualize_mesh:
        visualize_parameter_mesh.visualize_parameter_mesh(
            active_cells,
            particle_indexes=range(n),
            lower_bounds=bounds,
        )

    if visualize_search:
        if visualize_all_particles:
            for particle in range(n):
                draw_global_search(active_cells, bounds, particle=particle)

    if show_progress:
        elapsed = time.perf_counter() - start_time
        if processed_nodes > 0:
            seconds_per_1000 = (elapsed / processed_nodes) * 1000
            rate_text = f"{seconds_per_1000:.3f}s/1000 nodes"
        else:
            rate_text = "n/a s/1000 nodes"

        final_line = f"Progress: 100.0% (processed {processed_nodes} nodes, {rate_text})"
        progress_line_width = max(progress_line_width, len(final_line))
        print(final_line.ljust(progress_line_width))

    return best, best_config


def main():
    parser = argparse.ArgumentParser(description="Rigorous Taylor-search Thomson solver")
    parser.add_argument("n", nargs="?", type=int, default=4)
    parser.add_argument("--target-depth", type=int, default=12)
    parser.add_argument("--visualize-search", action="store_true")
    parser.add_argument("--visualize-all-particles", action="store_true")
    parser.add_argument("--visualize-mesh", action="store_true")
    parser.add_argument("--use-min-separation", action="store_true")
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument("--parallel-child-bounds", dest="parallel_child_bounds", action="store_true")
    parallel_group.add_argument("--no-parallel-child-bounds", dest="parallel_child_bounds", action="store_false")
    parser.set_defaults(parallel_child_bounds=True)
    parser.add_argument("--parallel-workers", type=int, default=mp_pool.cpu_count())
    parser.add_argument("--parallel-batch-size", type=int, default=32)
    parser.add_argument("--reuse-worker-pool", action="store_true")
    parser.add_argument("--measure-termination-volumes", action="store_true")
    parser.add_argument("--iv-dps", type=int, default=50)
    parser.add_argument("--initial-mesh-side-length", type=float, default=0.1)
    init_mode_group = parser.add_mutually_exclusive_group()
    init_mode_group.add_argument(
        "--antipodal",
        dest="initial_cell_mode",
        action="store_const",
        const="antipodal",
        help="Use antipodal initial-cell setup",
    )
    init_mode_group.add_argument(
        "--non-antipodal",
        dest="initial_cell_mode",
        action="store_const",
        const="non-antipodal",
        help="Use non-antipodal initial-cell setup (default)",
    )
    parser.add_argument("--no-visualize-final", action="store_true")
    parser.add_argument("--no-show-progress", action="store_true")
    parser.set_defaults(initial_cell_mode="non-antipodal")
    args = parser.parse_args()

    search(
        args.n,
        target_depth=args.target_depth,
        visualize_search=args.visualize_search,
        visualize_all_particles=args.visualize_all_particles,
        visualize_mesh=args.visualize_mesh,
        use_min_separation=args.use_min_separation,
        show_progress=not args.no_show_progress,
        parallel_child_bounds=args.parallel_child_bounds,
        parallel_workers=args.parallel_workers,
        parallel_batch_size=args.parallel_batch_size,
        reuse_worker_pool=args.reuse_worker_pool,
        measure_termination_volumes=args.measure_termination_volumes,
        iv_dps=args.iv_dps,
        initial_mesh_side_length=args.initial_mesh_side_length,
        initial_cell_mode=args.initial_cell_mode,
        visualize_final=not args.no_visualize_final,
    )


if __name__ == "__main__":
    main()
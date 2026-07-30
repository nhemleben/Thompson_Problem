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

from geometry import spherical_to_cart
from inital_part import initial_cell
from partition import split_with_index
from energy import thompson_energy
from search import (
    _center_config,
    _min_separation_cell_possible,
    _ordered_theta_center,
    _ordered_theta_possible,
    _respects_min_separation,
)
from visualizations import visualize_final_minimum, visualize_parameter_mesh
from visualizations.global_visualize import draw_global_search

mp.iv.dps = 50
sp_any: Any = sp
_TAYLOR_CACHE_VERSION = 1


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


def _tensor_frobenius_norm_bound(tensor) -> Any:
    array = np.asarray(tensor, dtype=object)
    total = mp.iv.mpf(0)

    for index in np.ndindex(array.shape):
        total += array[index] ** 2

    lower = max(0.0, float(total.a))
    upper = float(total.b)
    return mp.iv.sqrt(_iv_interval(lower, upper))


def _interval_cartesian_point(theta_bounds, phi_bounds):
    theta = _iv_interval(theta_bounds.lo, theta_bounds.hi)
    phi = _iv_interval(phi_bounds.lo, phi_bounds.hi)

    return (
        mp.iv.sin(phi) * mp.iv.cos(theta),  # type: ignore[operator]
        mp.iv.sin(phi) * mp.iv.sin(theta),  # type: ignore[operator]
        mp.iv.cos(phi),
    )


def _pair_distance_lower_bound(bounds_a, bounds_b) -> float:
    x1, y1, z1 = _interval_cartesian_point(*bounds_a)
    x2, y2, z2 = _interval_cartesian_point(*bounds_b)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2  # type: ignore[operator]

    distance_sq = dx * dx + dy * dy + dz * dz
    return math.sqrt(max(0.0, float(distance_sq.a)))


def _fallback_third_derivative_bound(cell) -> float:
    pairwise_bound = 0.0
    bounds = [particle_range.bounds for particle_range in cell.particle_ranges]

    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            d_min = _pair_distance_lower_bound(bounds[i], bounds[j])
            if d_min <= 0.0:
                return float("inf")

            pairwise_bound += 256.0 / (d_min ** 4)

    return pairwise_bound


def _taylor_enclosure(cell, model: TaylorModel):
    center, radius, intervals = _flat_center_and_radius(cell)
    config = _center_config(cell)

    qc_energy: Any = _iv_interval(thompson_energy(config), thompson_energy(config))
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


def _cascading_taylor_lower_bound(cell, model: TaylorModel, best_energy: float | None = None):
    center, radius, intervals = _flat_center_and_radius(cell)
    config = _center_config(cell)
    center_energy = thompson_energy(config)
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
        return None

    dq = [_iv_interval(-r, r) for r in radius]
    e0: Any = _iv_interval(center_energy, center_energy)

    # 2) Slightly more expensive: linear Taylor bound
    linear: Any = mp.iv.mpf(0)
    for gi, dqi in zip(gradient, dq):
        linear += gi * dqi

    linear_interval: Any = e0 + linear
    linear_lb = float(linear_interval.a)

    if best_energy is not None and linear_lb >= best_energy:
        return None

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
        return None

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
    return float(cubic_interval.a)


def _taylor_lower_bound(cell, model: TaylorModel) -> float:
    lower_bound = _cascading_taylor_lower_bound(cell, model, best_energy=None)
    if lower_bound is None:
        return float("inf")
    return lower_bound


def _child_taylor_lb_task(args):
    n, child, iv_dps, best_energy = args
    _set_iv_dps(iv_dps)
    model = build_taylor_model(n)
    return _cascading_taylor_lower_bound(child, model, best_energy=best_energy)


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
    parallel_workers=None,
    parallel_batch_size=64,
    iv_dps=50,
    visualize_final=True,
    d_min=None,
    alpha_min=None,
):
    iv_dps = _set_iv_dps(iv_dps)
    model = build_taylor_model(n)
    root = initial_cell(n)
    tie_breaker = count()

    queue = []
    active_cells = []
    bounds = []

    root_lb = _taylor_lower_bound(root, model)
    heapq.heappush(queue, (root_lb, next(tie_breaker), root))

    best = float("inf")
    best_config = None

    use_min_separation = (d_min is not None) or (alpha_min is not None)
    min_sep_cos_alpha = float(np.cos(alpha_min)) if alpha_min is not None else None
    min_sep_d_sq = float(d_min) * float(d_min) if d_min is not None else None

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
    if parallel_child_bounds:
        pool = mp_pool.Pool(processes=parallel_workers)

    try:
        use_batched_parallel = parallel_child_bounds and parallel_batch_size > 1

        while queue:
            if use_batched_parallel:
                batch_count = min(parallel_batch_size, len(queue))
                frontier = [heapq.heappop(queue) for _ in range(batch_count)]

                pending_children = []
                pending_tasks = []

                for lb, _, cell in frontier:
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

                    if use_min_separation:
                        if not _min_separation_cell_possible(
                            cell,
                            d_min=d_min,
                            alpha_min=alpha_min,
                            cos_alpha_min=min_sep_cos_alpha,
                            d_min_sq=min_sep_d_sq,
                        ):
                            continue

                    config = _center_config(cell)

                    refreshed_lb = _cascading_taylor_lower_bound(
                        cell,
                        model,
                        best_energy=best,
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
                        )

                    if center_feasible:
                        exact_energy = thompson_energy(config)
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
                        children, _ = split_with_index(cell)
                        children = [
                            child for child in children if _ordered_theta_possible(child)
                        ]

                        if not children:
                            continue

                        pending_children.extend(children)
                        pending_tasks.extend(
                            (n, child, int(iv_dps), best) for child in children
                        )

                if pending_tasks:
                    child_lbs = _evaluate_child_lb_tasks(pending_tasks, pool=pool)

                    for child, child_lb in zip(pending_children, child_lbs):
                        if child_lb is not None and child_lb < best:
                            heapq.heappush(queue, (child_lb, next(tie_breaker), child))
            else:
                lb, _, cell = heapq.heappop(queue)
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

                if use_min_separation:
                    if not _min_separation_cell_possible(
                        cell,
                        d_min=d_min,
                        alpha_min=alpha_min,
                        cos_alpha_min=min_sep_cos_alpha,
                        d_min_sq=min_sep_d_sq,
                    ):
                        continue

                config = _center_config(cell)

                refreshed_lb = _cascading_taylor_lower_bound(
                    cell,
                    model,
                    best_energy=best,
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
                    )

                if center_feasible:
                    exact_energy = thompson_energy(config)
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
                    children, _ = split_with_index(cell)
                    children = [
                        child for child in children if _ordered_theta_possible(child)
                    ]

                    child_tasks = [(n, child, int(iv_dps), best) for child in children]
                    child_lbs = _evaluate_child_lb_tasks(child_tasks, pool=pool)

                    for child, child_lb in zip(children, child_lbs):
                        if child_lb is not None and child_lb < best:
                            heapq.heappush(queue, (child_lb, next(tie_breaker), child))

        if show_progress and not queue:
            _print_progress_line(processed_nodes)
    finally:
        if pool is not None:
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
    parser.add_argument("--parallel-child-bounds", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--parallel-batch-size", type=int, default=64)
    parser.add_argument("--iv-dps", type=int, default=50)
    parser.add_argument("--no-visualize-final", action="store_true")
    parser.add_argument("--no-show-progress", action="store_true")
    args = parser.parse_args()

    search(
        args.n,
        target_depth=args.target_depth,
        visualize_search=args.visualize_search,
        visualize_all_particles=args.visualize_all_particles,
        visualize_mesh=args.visualize_mesh,
        show_progress=not args.no_show_progress,
        parallel_child_bounds=args.parallel_child_bounds,
        parallel_workers=args.parallel_workers,
        parallel_batch_size=args.parallel_batch_size,
        iv_dps=args.iv_dps,
        visualize_final=not args.no_visualize_final,
    )


if __name__ == "__main__":
    main()
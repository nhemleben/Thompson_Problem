import argparse
import ast
import math
from statistics import mean

import numpy as np

import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geometry import spherical_to_cart
from last_particle_lipshitz import psi_lipschitz, psi_value, torque_lipschitz, torque_jacobian,spherical_jacobian
from taylor_search import _set_iv_dps
from derivative import spherical_jacobian as coordinate_jacobian, thomson_gradient
from energy import thompson_energy


def _parse_candidate(candidate_text: str) -> list[tuple[float, float]]:
    obj = ast.literal_eval(candidate_text)
    if not isinstance(obj, (list, tuple)):
        raise ValueError("Candidate must be a list of (theta, phi) pairs")

    candidate: list[tuple[float, float]] = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each candidate entry must be a (theta, phi) pair")
        candidate.append((float(item[0]), float(item[1])))

    return candidate


def _flat_config(config: list[tuple[float, float]]) -> np.ndarray:
    flat: list[float] = []
    for theta, phi in config:
        flat.append(float(theta))
        flat.append(float(phi))
    return np.asarray(flat, dtype=float)


def _abs_upper(value) -> float:
    if hasattr(value, "a") and hasattr(value, "b"):
        lo = float(value.a)
        hi = float(value.b)
        return max(abs(lo), abs(hi))
    return abs(float(value))


def _spherical_energy_gradient(config: list[tuple[float, float]]) -> np.ndarray:
    xyz = np.asarray([spherical_to_cart(*point) for point in config], dtype=float)
    cartesian_gradient = thomson_gradient(xyz)
    spherical_gradient = np.zeros(2 * len(config), dtype=float)

    for index, (theta, phi) in enumerate(config):
        jacobian = coordinate_jacobian(theta, phi)
        spherical_gradient[2 * index: 2 * index + 2] = jacobian.T @ cartesian_gradient[index]

    return spherical_gradient


def current_taylor_lipschitz(config: list[tuple[float, float]], cell_radius: float):
    gradient = _spherical_energy_gradient(config)
    grad_l1 = sum(_abs_upper(component) for component in gradient)
    energy = float(thompson_energy(config))

    return {
        "L_current": float(grad_l1),
        "bound_current": float(grad_l1) * float(cell_radius),
        "energy": energy,
        "energy/L_current": energy / grad_l1 if grad_l1 != 0 else math.inf,
        "energy/bound_current": energy / (grad_l1 * float(cell_radius)) if grad_l1 != 0 else math.inf,
    }


def last_particle_lipschitz_bounds(config: list[tuple[float, float]], cell_radius: float):
    if len(config) < 2:
        raise ValueError("Need at least 2 particles for last-particle comparison")

    x = spherical_to_cart(*config[-1])
    fixed_points = np.asarray([spherical_to_cart(*point) for point in config[:-1]], dtype=float)

    l_torque = float(torque_lipschitz(x, fixed_points, cell_radius))
    l_psi = float(psi_lipschitz(x, fixed_points, cell_radius))
    psi_at_center = float(psi_value(x, fixed_points))
    j_norm = float(np.linalg.norm(torque_jacobian(x, fixed_points), ord=2))
    ratio_jnorm_to_current = psi_at_center/j_norm if j_norm != 0 else math.inf
    spherical_jacob = spherical_jacobian(torque_jacobian(x, fixed_points), x)
    spherical_jacob_norm = float(np.linalg.norm(spherical_jacob, ord=2))

    return {
        "L_torque": l_torque,
        "L_psi": l_psi,
        "bound_psi": l_psi * float(cell_radius),
        "psi_center": psi_at_center,
        "psi/L_torque": psi_at_center / l_torque if l_torque != 0 else math.inf,
        "psi/L_psi": psi_at_center / l_psi if l_psi != 0 else math.inf,
        "j_norm": j_norm,
        "ratio_jnorm_to_current": ratio_jnorm_to_current,
        "spherical_jacob_norm": spherical_jacob_norm,
        "psi/spherical_jacob_norm": psi_at_center / spherical_jacob_norm if spherical_jacob_norm != 0 else math.inf
    }


def compare_lipschitz_bounds(config: list[tuple[float, float]], cell_radius: float):
    result = {}
    result.update(current_taylor_lipschitz(config, cell_radius))
    result.update(last_particle_lipschitz_bounds(config, cell_radius))

    if result["L_current"] > 0:
        result["ratio_Lpsi_to_current"] = result["L_psi"] / result["L_current"]
        result["ratio_Ltorque_to_current"] = result["L_torque"] / result["L_current"]
    else:
        result["ratio_Lpsi_to_current"] = math.inf
        result["ratio_Ltorque_to_current"] = math.inf

    if result["bound_current"] > 0:
        result["ratio_boundpsi_to_current"] = result["bound_psi"] / result["bound_current"]
    else:
        result["ratio_boundpsi_to_current"] = math.inf
    if result["j_norm"] > 0:
        result["ratio_jnorm_to_current"] = result["j_norm"] / result["L_current"]
    else:
        result["ratio_jnorm_to_current"] = math.inf
    if result["spherical_jacob_norm"] > 0:
        result["ratio_spherical_jacob_to_current"] = result["spherical_jacob_norm"] / result["L_current"]
    else:
        result["ratio_spherical_jacob_to_current"] = math.inf

    return result


def random_configuration(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    config: list[tuple[float, float]] = []

    for _ in range(n):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        u = float(rng.uniform(-1.0, 1.0))
        phi = float(math.acos(u))
        config.append((theta, phi))

    return config


def compare_random_configurations(k: int, n: int, cell_radius: float, seed: int | None = None):
    if k <= 0:
        raise ValueError("k must be positive")

    rng = np.random.default_rng(seed)
    results = []

    for _ in range(k):
        config = random_configuration(n, rng)
        results.append(compare_lipschitz_bounds(config, cell_radius))

    summary = {
        "k": k,
        "n": n,
        "cell_radius": cell_radius,
        "avg_L_current": mean(r["L_current"] for r in results),
        "avg_L_torque": mean(r["L_torque"] for r in results),
        "avg_L_psi": mean(r["L_psi"] for r in results),
        "avg_bound_current": mean(r["bound_current"] for r in results),
        "avg_bound_psi": mean(r["bound_psi"] for r in results),
        "avg_psi_center": mean(r["psi_center"] for r in results),
        "avg_ratio_Lpsi_to_current": mean(r["ratio_Lpsi_to_current"] for r in results),
        "avg_ratio_Ltorque_to_current": mean(r["ratio_Ltorque_to_current"] for r in results),
        "avg_ratio_boundpsi_to_current": mean(r["ratio_boundpsi_to_current"] for r in results),
        "min_ratio_Lpsi_to_current": min(r["ratio_Lpsi_to_current"] for r in results),
        "max_ratio_Lpsi_to_current": max(r["ratio_Lpsi_to_current"] for r in results),
        "min_psi_center": min(r["psi_center"] for r in results),
        "max_psi_center": max(r["psi_center"] for r in results),
        "avg_energy": mean(r["energy"] for r in results),
        "min_energy": min(r["energy"] for r in results),
        "max_energy": max(r["energy"] for r in results),
        "avg_energy/L_current": mean(r["energy/L_current"] for r in results),
        "avg_psi/L_psi": mean(r["psi/L_psi"] for r in results),
        "avg_psi/L_torque": mean(r["psi/L_torque"] for r in results),
        "avg_j_norm": mean(r["j_norm"] for r in results),
        "avg_psi/j_norm": mean(r["ratio_jnorm_to_current"] for r in results),
        "avg_psi/spherical_jacob_norm": mean(r["psi/spherical_jacob_norm"] for r in results),
        "avg_spherical_jacob_norm": mean(r["spherical_jacob_norm"] for r in results),
        "avg_spherical_jacob_to_current": mean(r["ratio_spherical_jacob_to_current"] for r in results),
    }

    return summary, results


def _print_single(result: dict):
    print("Single-Configuration Comparison")
    print(f"  L_current (taylor_search): {result['L_current']:.12e}")
    print(f"  bound_current = L_current * r: {result['bound_current']:.12e}")
    print(f"  L_torque (last_particle_lipshitz): {result['L_torque']:.12e}")
    print(f"  L_psi (last_particle_lipshitz): {result['L_psi']:.12e}")
    print(f"  bound_psi = L_psi * r: {result['bound_psi']:.12e}")
    print(f"  psi(center): {result['psi_center']:.12e}")
    print(f"  L_psi / L_current: {result['ratio_Lpsi_to_current']:.12e}")
    print(f"  L_torque / L_current: {result['ratio_Ltorque_to_current']:.12e}")
    print(f"  bound_psi / bound_current: {result['ratio_boundpsi_to_current']:.12e}")


def _print_random_summary(summary: dict):
    print("Random-Configuration Comparison Summary")
    print(f"  k: {summary['k']}")
    print(f"  n: {summary['n']}")
    print(f"  cell_radius: {summary['cell_radius']}")
    print(f"  avg L_current: {summary['avg_L_current']:.12e}")
    print(f"  avg L_torque: {summary['avg_L_torque']:.12e}")
    print(f"  avg L_psi: {summary['avg_L_psi']:.12e}")
#    print(f"  avg bound_current: {summary['avg_bound_current']:.12e}")
#    print(f"  avg bound_psi: {summary['avg_bound_psi']:.12e}")
#    print(f"  avg (L_psi / L_current): {summary['avg_ratio_Lpsi_to_current']:.12e}")
#    print(f"  avg (L_torque / L_current): {summary['avg_ratio_Ltorque_to_current']:.12e}")
#    print(f"  avg (bound_psi / bound_current): {summary['avg_ratio_boundpsi_to_current']:.12e}")
#    print(f"  min (L_psi / L_current): {summary['min_ratio_Lpsi_to_current']:.12e}")
#    print(f"  max (L_psi / L_current): {summary['max_ratio_Lpsi_to_current']:.12e}")
    print(f"  avg psi(center): {summary['avg_psi_center']:.12e}")
    print(f"  min psi(center): {summary['min_psi_center']:.12e}")
    print(f"  max psi(center): {summary['max_psi_center']:.12e}")
    print(f"  avg energy: {summary['avg_energy']:.12e}")
    print(f"  min energy: {summary['min_energy']:.12e}")
    print(f"  max energy: {summary['max_energy']:.12e}")
    print(f"  avg (energy / L_current): {summary['avg_energy/L_current']:.12e}")
    print(f"  avg (psi / L_psi): {summary['avg_psi/L_psi']:.12e}")
    print(f"  avg (psi / L_torque): {summary['avg_psi/L_torque']:.12e}")
    print(f"  avg (psi / torque_jacobian): {summary['avg_psi/j_norm']:.12e}")
    print(f"  avg (psi/ spherical_jacob_norm): {summary['avg_psi/spherical_jacob_norm']:.12e}")
    print(f"  avg spherical_jacob_norm: {summary['avg_spherical_jacob_norm']:.12e}")
    print(f"  avg (spherical_jacob / L_current): {summary['avg_spherical_jacob_to_current']:.12e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current Taylor-search Lipschitz magnitude against "
            "psi/torque Lipschitz bounds for the last-particle test."
        )
    )
    parser.add_argument("--n", type=int, default=4, help="Number of particles")
    parser.add_argument("--cell-radius", type=float, default=1e-2, help="Cell radius used in bounds")
    parser.add_argument("--iv-dps", type=int, default=40, help="Interval precision for Taylor model")
    parser.add_argument(
        "--candidate",
        type=str,
        default="",
        help="Optional candidate as list of (theta, phi) pairs",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="If > 0, run comparison over k random configurations",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --k mode")
    args = parser.parse_args()

    _set_iv_dps(int(args.iv_dps))

    if args.candidate:
        config = _parse_candidate(args.candidate)
        result = compare_lipschitz_bounds(config, args.cell_radius)
        _print_single(result)

    if args.k > 0:
        summary, _ = compare_random_configurations(
            k=int(args.k),
            n=int(args.n),
            cell_radius=float(args.cell_radius),
            seed=args.seed,
        )
        _print_random_summary(summary)

    if not args.candidate and args.k <= 0:
        rng = np.random.default_rng(args.seed)
        config = random_configuration(int(args.n), rng)
        result = compare_lipschitz_bounds(config, args.cell_radius)
        _print_single(result)


if __name__ == "__main__":
    main()

import argparse
import math
from statistics import mean

import numpy as np

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from known_optimal import T_n
from last_particle_centroid_bound import last_particle_cap, last_particle_cap_tighter


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return random_unit_vector(rng)
    return vector / norm


def random_cartesian_configuration(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    return np.asarray([random_unit_vector(rng) for _ in range(n)], dtype=float)


def last_particle_energy(points: np.ndarray) -> float:
    if len(points) < 2:
        raise ValueError("Need at least 2 points to compute the last-particle energy")

    last_point = points[-1]
    energy = 0.0
    for point in points[:-1]:
        distance = float(np.linalg.norm(point - last_point))
        if distance == 0.0:
            return math.inf
        energy += 1.0 / distance
    return energy


def total_pairwise_energy(points: np.ndarray) -> float:
    energy = 0.0
    for index in range(len(points)):
        for other_index in range(index + 1, len(points)):
            distance = float(np.linalg.norm(points[index] - points[other_index]))
            if distance == 0.0:
                return math.inf
            energy += 1.0 / distance
    return energy


def cap_area_from_result(result: dict) -> float:
    sphere_area = 4.0 * math.pi

    if result.get("all_sphere"):
        return sphere_area

    if result.get("empty"):
        return 0.0

    alpha = float(result["angular_radius"])
    return 2.0 * math.pi * (1.0 - math.cos(alpha))


def compare_random_centroid_bounds(k: int, n: int, seed: int | None = None):
    if k <= 0:
        raise ValueError("k must be positive")

    rng = np.random.default_rng(seed)
    results = []

    for _ in range(k):
        points = random_cartesian_configuration(n, rng)
        full_configuration_energy = total_pairwise_energy(points)
        nth_particle_energy = last_particle_energy(points)
        E_prev = full_configuration_energy - nth_particle_energy

        #Use known optimal n-1 and n configuration energy
        U_n = T_n(n)
        E_prev = T_n(n - 1)
        cap = last_particle_cap_tighter(points, U_n=U_n, E_prev=E_prev)
        area = cap_area_from_result(cap)

        results.append(
            {
                "points": points,
                "cap": cap,
                "cap_area": area,
            }
        )

    sphere_area = 4.0 * math.pi
    avg_cap_area = mean(item["cap_area"] for item in results)
    percent_retained = 100.0 * avg_cap_area / sphere_area
    percent_eliminated = 100.0 - percent_retained

    summary = {
        "k": k,
        "n": n,
        "avg_cap_area": avg_cap_area,
        "sphere_area": sphere_area,
        "percent_eliminated": percent_eliminated,
        "percent_retained": percent_retained,
        "min_cap_area": min(item["cap_area"] for item in results),
        "max_cap_area": max(item["cap_area"] for item in results),
        "avg_angular_radius": mean(
            float(item["cap"]["angular_radius"])
            for item in results
            if "angular_radius" in item["cap"]
        ) if any("angular_radius" in item["cap"] for item in results) else 0.0,
        "all_sphere_count": sum(1 for item in results if item["cap"].get("all_sphere")),
        "empty_count": sum(1 for item in results if item["cap"].get("empty")),
    }

    return summary, results


def _print_summary(summary: dict):
    print("Centroid Bound Eliminated Area Summary")
    print(f"  k: {summary['k']}")
    print(f"  n: {summary['n']}")
    print(f"  average cap area: {summary['avg_cap_area']:.12e}")
    print(f"  full sphere area: {summary['sphere_area']:.12e}")
    print(f"  percentage eliminated: {summary['percent_eliminated']:.12f}%")
    print(f"  percentage retained: {summary['percent_retained']:.12f}%")
    print(f"  min cap area: {summary['min_cap_area']:.12e}")
    print(f"  max cap area: {summary['max_cap_area']:.12e}")
    print(f"  average angular radius: {summary['avg_angular_radius']:.12e}")
    print(f"  all-sphere cases: {summary['all_sphere_count']}")
    print(f"  empty cases: {summary['empty_count']}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample random n-particle configurations, estimate the last-particle "
            "allowable cap area, and report how much of the sphere is eliminated."
        )
    )
    parser.add_argument("--n", type=int, default=4, help="Number of particles")
    parser.add_argument("--k", type=int, default=100, help="Number of random configurations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    summary, _ = compare_random_centroid_bounds(
        k=int(args.k),
        n=int(args.n),
        seed=args.seed,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
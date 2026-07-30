from pathlib import Path
import argparse
import contextlib
import importlib
import io
import statistics
import sys
import time

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

search = importlib.import_module("taylor_search").search


def parse_precisions(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))

    if not values:
        raise ValueError("At least one precision is required")

    return values


def run_once(n: int, depth: int, iv_dps: int):
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        energy, config = search(
            n,
            target_depth=depth,
            show_progress=False,
            visualize_search=False,
            visualize_mesh=False,
            visualize_final=False,
            iv_dps=iv_dps,
        )
    elapsed = time.perf_counter() - start
    return elapsed, energy, len(config)


def summarize(iv_dps: int, times: list[float], energy: float, points: int):
    avg = statistics.mean(times)
    med = statistics.median(times)
    best = min(times)
    worst = max(times)
    print(
        f"iv_dps={iv_dps:>3} | avg={avg:8.4f}s | med={med:8.4f}s | "
        f"best={best:8.4f}s | worst={worst:8.4f}s | points={points} | E={energy:.12f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Taylor search versus mpmath interval precision"
    )
    parser.add_argument("--n", type=int, default=5, help="Number of particles")
    parser.add_argument("--depth", type=int, default=8, help="Target search depth")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per precision")
    parser.add_argument(
        "--precisions",
        type=str,
        default="30,40,50",
        help="Comma-separated iv_dps values",
    )
    args = parser.parse_args()

    precision_values = parse_precisions(args.precisions)

    print("Taylor Precision Benchmark")
    print(f"  n={args.n}, depth={args.depth}, repeats={args.repeats}")
    print(f"  iv_dps values={precision_values}")
    print()

    for iv_dps in precision_values:
        times: list[float] = []
        last_energy = 0.0
        last_points = 0

        for _ in range(args.repeats):
            elapsed, energy, points = run_once(args.n, args.depth, iv_dps)
            times.append(elapsed)
            last_energy = energy
            last_points = points

        summarize(iv_dps, times, last_energy, last_points)


if __name__ == "__main__":
    main()

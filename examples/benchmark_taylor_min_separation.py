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
bound_module = importlib.import_module("bound")


"""
Benchmark Taylor search with min-separation pruning off vs on.
"""
def run_once(
    n: int,
    depth: int,
    iv_dps: int,
    use_min_separation: bool,
    parallel_child_bounds: bool,
    parallel_workers: int | None,
    parallel_batch_size: int,
):
    d_min_value = None
    alpha_min_value = None

    if use_min_separation:
        d_min_value = float(bound_module.d_min(n))
        alpha_min_value = float(bound_module.angle_min(n))

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
            d_min=d_min_value,
            alpha_min=alpha_min_value,
            parallel_child_bounds=parallel_child_bounds,
            parallel_workers=parallel_workers,
            parallel_batch_size=parallel_batch_size,
        )
    elapsed = time.perf_counter() - start
    point_count = len(config) if config is not None else 0
    return elapsed, energy, point_count


def summarize(label: str, times: list[float], energy: float, points: int):
    avg = statistics.mean(times)
    med = statistics.median(times)
    best = min(times)
    worst = max(times)
    print(
        f"{label:>18} | avg={avg:8.4f}s | med={med:8.4f}s | "
        f"best={best:8.4f}s | worst={worst:8.4f}s | points={points} | E={energy:.12f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Taylor search with min-separation pruning off vs on"
    )
    parser.add_argument("--n", type=int, default=5, help="Number of particles")
    parser.add_argument("--depth", type=int, default=8, help="Target search depth")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per mode")
    parser.add_argument("--iv-dps", type=int, default=40, help="mpmath interval precision")
    parser.add_argument("--parallel-child-bounds", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--parallel-batch-size", type=int, default=64)
    args = parser.parse_args()

    print("Taylor Min-Separation Benchmark")
    print(f"  n={args.n}, depth={args.depth}, repeats={args.repeats}, iv_dps={args.iv_dps}")
    print(
        "  parallel_child_bounds="
        f"{args.parallel_child_bounds}, parallel_workers={args.parallel_workers}, "
        f"parallel_batch_size={args.parallel_batch_size}"
    )
    print()

    off_times: list[float] = []
    off_energy = 0.0
    off_points = 0

    for _ in range(args.repeats):
        elapsed, energy, points = run_once(
            args.n,
            args.depth,
            args.iv_dps,
            use_min_separation=False,
            parallel_child_bounds=args.parallel_child_bounds,
            parallel_workers=args.parallel_workers,
            parallel_batch_size=args.parallel_batch_size,
        )
        off_times.append(elapsed)
        off_energy = energy
        off_points = points

    on_times: list[float] = []
    on_energy = 0.0
    on_points = 0

    for _ in range(args.repeats):
        elapsed, energy, points = run_once(
            args.n,
            args.depth,
            args.iv_dps,
            use_min_separation=True,
            parallel_child_bounds=args.parallel_child_bounds,
            parallel_workers=args.parallel_workers,
            parallel_batch_size=args.parallel_batch_size,
        )
        on_times.append(elapsed)
        on_energy = energy
        on_points = points

    summarize("min_sep OFF", off_times, off_energy, off_points)
    summarize("min_sep ON", on_times, on_energy, on_points)

    off_avg = statistics.mean(off_times)
    on_avg = statistics.mean(on_times)

    if off_avg > 0.0:
        ratio = on_avg / off_avg
        delta = on_avg - off_avg
        print()
        print(
            f"Relative: ON/OFF = {ratio:.3f}x, "
            f"delta = {delta:+.4f}s ({(ratio - 1.0) * 100:+.1f}%)"
        )


if __name__ == "__main__":
    main()

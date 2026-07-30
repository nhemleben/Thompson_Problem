from pathlib import Path
import sys
import os
import io
import time
import argparse
import importlib
import contextlib
import statistics

import matplotlib

matplotlib.use("Agg")

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

search = importlib.import_module("taylor_search").search


def run_search_once(
    n,
    target_depth,
    parallel_enabled,
    workers,
    batch_size,
    iv_dps,
    initial_mesh_side_length,
):
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        energy, config = search(
            n,
            target_depth=target_depth,
            visualize_search=False,
            visualize_mesh=False,
            visualize_final=False,
            show_progress=False,
            parallel_child_bounds=parallel_enabled,
            parallel_workers=workers,
            parallel_batch_size=batch_size,
            iv_dps=iv_dps,
            initial_mesh_side_length=initial_mesh_side_length,
        )
    elapsed = time.perf_counter() - start
    point_count = len(config) if config is not None else 0
    return elapsed, energy, point_count


def summarize(label, times, energy):
    avg = statistics.mean(times)
    med = statistics.median(times)
    best = min(times)
    worst = max(times)
    print(
        f"{label:>18} | avg={avg:8.4f}s | med={med:8.4f}s | "
        f"best={best:8.4f}s | worst={worst:8.4f}s | E={energy:.12f}"
    )


def parse_batch_sizes(text):
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one batch size is required")
    return values


def main():
    parser = argparse.ArgumentParser(description="Benchmark taylor_search parallel batch sizes")
    parser.add_argument("--n", type=int, default=5, help="Number of particles")
    parser.add_argument("--depth", type=int, default=2, help="Single search target depth")
    parser.add_argument(
        "--depths",
        type=str,
        default="",
        help="Optional comma-separated depth sweep, e.g. 10,12,14,16",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per configuration")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Worker process count for parallel runs",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="16,32,64,128,256,512,1024,2048",
        help="Comma-separated parallel_batch_size values",
    )
    parser.add_argument("--iv-dps", type=int, default=40, help="mpmath interval precision")
    parser.add_argument("--initial-mesh-side-length", type=float, default=2)
    args = parser.parse_args()

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    depth_values = parse_batch_sizes(args.depths) if args.depths else [args.depth]

    print("Taylor Parallel Batch Benchmark")
    print(f"  n={args.n}, depths={depth_values}, repeats={args.repeats}, workers={args.workers}")
    print(f"  batch_sizes={batch_sizes}")
    print(
        f"  iv_dps={args.iv_dps}, initial_mesh_side_length={args.initial_mesh_side_length}"
    )
    print()

    for depth in depth_values:
        print(f"Depth {depth}")

        serial_times = []
        serial_energy = None
        for _ in range(args.repeats):
            elapsed, energy, _ = run_search_once(
                args.n,
                depth,
                parallel_enabled=False,
                workers=None,
                batch_size=32,
                iv_dps=args.iv_dps,
                initial_mesh_side_length=args.initial_mesh_side_length,
            )
            serial_times.append(elapsed)
            serial_energy = energy
            print(f"  Serial run: elapsed={elapsed:.4f}s, energy={energy:.12f}")

        summarize("serial", serial_times, serial_energy)

        for batch_size in batch_sizes:
            parallel_times = []
            parallel_energy = None
            for _ in range(args.repeats):
                elapsed, energy, _ = run_search_once(
                    args.n,
                    depth,
                    parallel_enabled=True,
                    workers=args.workers,
                    batch_size=batch_size,
                    iv_dps=args.iv_dps,
                    initial_mesh_side_length=args.initial_mesh_side_length,
                )
                parallel_times.append(elapsed)
                parallel_energy = energy
                print(f"  Parallel run (b={batch_size}): elapsed={elapsed:.4f}s, energy={energy:.12f}")

            summarize(f"parallel b={batch_size}", parallel_times, parallel_energy)

        print()


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import os
import io
import time
import argparse
import contextlib
import statistics

import matplotlib

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search


def run_search_once(n, target_depth, parallel_enabled, workers, batch_size):
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
        )
    elapsed = time.perf_counter() - start
    return elapsed, energy, len(config)


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
    parser = argparse.ArgumentParser(description="Benchmark search parallel batch sizes")
    parser.add_argument("--n", type=int, default=6, help="Number of particles")
    parser.add_argument("--depth", type=int, default=12, help="Single search target depth")
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
    args = parser.parse_args()

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    depth_values = parse_batch_sizes(args.depths) if args.depths else [args.depth]

    print("Benchmark settings")
    print(f"  n={args.n}, depths={depth_values}, repeats={args.repeats}, workers={args.workers}")
    print(f"  batch_sizes={batch_sizes}")
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
                batch_size=64,
            )
            serial_times.append(elapsed)
            serial_energy = energy

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
                )
                parallel_times.append(elapsed)
                parallel_energy = energy

            summarize(f"parallel b={batch_size}", parallel_times, parallel_energy)

        print()


if __name__ == "__main__":
    main()

import argparse
import time
from pathlib import Path

from taylor_search import materialize_symbolic_derivatives


def parse_n_values(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(int(item))

    if not values:
        raise ValueError("No n values were provided")

    return values


def main():
    parser = argparse.ArgumentParser(
        description="Generate and store symbolic derivative payloads for taylor_search"
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="",
        help="Comma-separated n values, e.g. 3,4,5",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=3,
        help="Minimum n for range mode",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=6,
        help="Maximum n for range mode (inclusive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild files even if they already exist",
    )
    args = parser.parse_args()

    if args.n_values:
        n_values = parse_n_values(args.n_values)
    else:
        if args.n_min > args.n_max:
            raise ValueError("--n-min must be <= --n-max")
        n_values = list(range(args.n_min, args.n_max + 1))

    total_start = time.perf_counter()
    print(f"Generating symbolic derivative payloads for n={n_values}")

    for n in n_values:
        start = time.perf_counter()
        file_path = materialize_symbolic_derivatives(n, force=args.force)
        elapsed = time.perf_counter() - start
        size_kb = file_path.stat().st_size / 1024.0
        print(f"n={n}: {file_path} ({size_kb:.1f} KB) in {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    print(f"Done in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

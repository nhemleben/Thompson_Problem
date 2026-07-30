import argparse
import time

from taylor_search import materialize_symbolic_derivatives


def main():
    parser = argparse.ArgumentParser(
        description="Generate symbolic derivative payloads for all n from 3 up to a limit"
    )
    parser.add_argument(
        "--n-max",
        type=int,
        required=True,
        help="Maximum n value to generate (inclusive)",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=3,
        help="Minimum n value to generate (inclusive)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even if they already exist",
    )
    args = parser.parse_args()

    if args.n_min < 1:
        raise ValueError("--n-min must be >= 1")

    if args.n_max < args.n_min:
        raise ValueError("--n-max must be >= --n-min")

    n_values = list(range(args.n_min, args.n_max + 1))

    total_start = time.perf_counter()
    print(f"Generating symbolic derivatives for n={args.n_min}..{args.n_max}")

    for n in n_values:
        start = time.perf_counter()
        file_path = materialize_symbolic_derivatives(n, force=args.force)
        elapsed = time.perf_counter() - start
        size_kb = file_path.stat().st_size / 1024.0
        print(f"n={n}: {file_path} ({size_kb:.1f} KB) in {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    print(f"Completed {len(n_values)} files in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import importlib
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

search = importlib.import_module("taylor_search").search


def main():
    parser = argparse.ArgumentParser(description="Run the rigorous Taylor-search Thomson solver")
    parser.add_argument("--n", type=int, default=3, help="Number of particles")
    parser.add_argument("--target-depth", type=int, default=12, help="Search tree depth")
    parser.add_argument("--iv-dps", type=int, default=50, help="mpmath interval precision")
    parser.add_argument(
        "--visualize-search",
        action="store_true",
        help="Plot the active search cells while running",
    )
    parser.add_argument(
        "--visualize-final",
        action="store_true",
        help="Show the final minimum configuration",
    )
    parser.add_argument(
        "--no-show-progress",
        action="store_true",
        help="Disable progress output",
    )
    args = parser.parse_args()

    energy, config = search(
        args.n,
        target_depth=args.target_depth,
        visualize_search=args.visualize_search,
        visualize_final=args.visualize_final,
        iv_dps=args.iv_dps,
        show_progress=not args.no_show_progress,
    )

    print(energy, f"Taylor-search result for n={args.n}")
    print(config)


if __name__ == "__main__":
    main(
    )

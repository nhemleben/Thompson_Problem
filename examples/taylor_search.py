from pathlib import Path
import argparse
import importlib
import multiprocessing as mp
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

search = importlib.import_module("taylor_search").search


def main():
    parser = argparse.ArgumentParser(description="Run the rigorous Taylor-search Thomson solver")
    parser.add_argument("--n", type=int, default=3, help="Number of particles")
    parser.add_argument("--target-depth", type=int, default=0, help="Search depth")
    parser.add_argument("--iv-dps", type=int, default=50, help="mpmath interval precision")
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--parallel-child-bounds",
        dest="parallel_child_bounds",
        action="store_true",
        help="Enable parallel child lower-bound evaluation",
    )
    parallel_group.add_argument(
        "--no-parallel-child-bounds",
        dest="parallel_child_bounds",
        action="store_false",
        help="Disable parallel child lower-bound evaluation",
    )
    parser.set_defaults(parallel_child_bounds=True)
    parser.add_argument("--parallel-workers", type=int, default=mp.cpu_count())
    parser.add_argument("--parallel-batch-size", type=int, default=32)
    parser.add_argument("--initial-mesh-side-length", type=float, default=0.1)
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
        parallel_child_bounds=args.parallel_child_bounds,
        parallel_workers=args.parallel_workers,
        parallel_batch_size=args.parallel_batch_size,
        initial_mesh_side_length=args.initial_mesh_side_length,
        show_progress=not args.no_show_progress,
    )

    print(energy)
    print(config)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
from itertools import combinations
from typing import Any
import argparse

import matplotlib.pyplot as plt
import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1]))

from geometry import spherical_to_cart
from visualizations.visualize import draw_sphere


def draw_full_mesh(ax, xyz):
    for start, end in combinations(xyz, 2):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="blue",
            linewidth=1.25,
            alpha=0.85,
        )


def draw_nearest_neighbor_mesh(ax, xyz):
    edges = set()
    for index, start in enumerate(xyz):
        nearest_index = None
        nearest_distance = float("inf")

        for other_index, end in enumerate(xyz):
            if index == other_index:
                continue

            distance = np.linalg.norm(start - end)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = other_index

        if nearest_index is not None:
            edges.add(tuple(sorted((index, nearest_index))))

    for start_index, end_index in sorted(edges):
        start = xyz[start_index]
        end = xyz[end_index]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="blue",
            linewidth=1.25,
            alpha=0.85,
        )


def plot_final_minimum(points, energy, use_nearest_neighbors=False):
    fig = plt.figure(figsize=(8, 8))
    ax: Any = fig.add_subplot(111, projection="3d")

    draw_sphere(ax)

    xyz = np.array([spherical_to_cart(theta, phi) for theta, phi in points])

    if use_nearest_neighbors:
        draw_nearest_neighbor_mesh(ax, xyz)
    else:
        draw_full_mesh(ax, xyz)

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        **{"s": 80},
        c=np.linspace(0, 1, len(xyz)),
        cmap="viridis",
        depthshade=True,
    )

    for index, (x, y, z) in enumerate(xyz, start=1):
        ax.text(x, y, z, s=f"  {index}")

    ax.set_title(f"Final minimum configuration, E = {energy:.6f}")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the final minimum configuration found by the search"
    )
    parser.add_argument(
        "--nearest-neighbors",
        action="store_true",
        help="Draw only nearest-neighbor connections instead of the full mesh",
    )
    args = parser.parse_args()

    from search import search

    n = 3
    target_depth = 12

    energy, config = search(n, target_depth=target_depth, visualize_search=False)
    plot_final_minimum(
        config,
        energy,
        use_nearest_neighbors=args.nearest_neighbors,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Analyze surviving Thomson-problem search cells."
    )

    parser.add_argument(
        "--input",
        help="Input TXT/CSV file"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output sorted CSV file"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins (default: 100)"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of lowest-energy points to print (default: 20)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Read file
    # ------------------------------------------------------------

    df = pd.read_csv(args.input)

    if "center_energy" not in df.columns:
        raise ValueError(
            "Could not find 'center_energy' column.\n"
            f"Columns found: {list(df.columns)}"
        )

    if "gradient_norm" not in df.columns:
        raise ValueError(
            "Could not find 'gradient_norm' column.\n"
            f"Columns found: {list(df.columns)}"
        )

    # Convert to numeric in case some values were read as strings.
    df["center_energy"] = pd.to_numeric(
        df["center_energy"],
        errors="coerce"
    )

    df["gradient_norm"] = pd.to_numeric(
        df["gradient_norm"],
        errors="coerce"
    )

    # ------------------------------------------------------------
    # Energy statistics
    # ------------------------------------------------------------

    energy = df["center_energy"].to_numpy()

    inf_mask = np.isinf(energy)
    nan_mask = np.isnan(energy)
    finite_mask = np.isfinite(energy)

    n_total = len(df)
    n_inf = np.count_nonzero(inf_mask)
    n_pos_inf = np.count_nonzero(np.isposinf(energy))
    n_neg_inf = np.count_nonzero(np.isneginf(energy))
    n_nan = np.count_nonzero(nan_mask)
    n_finite = np.count_nonzero(finite_mask)

    print()
    print("=" * 60)
    print("Thomson Search-Cell Statistics")
    print("=" * 60)

    print(f"Total points/cells:       {n_total}")
    print(f"Finite energy:            {n_finite}")
    print(f"Inf energy:               {n_inf}")
    print(f"    +inf:                 {n_pos_inf}")
    print(f"    -inf:                 {n_neg_inf}")
    print(f"NaN energy:               {n_nan}")

    if n_finite == 0:
        print("\nNo finite-energy points to analyze.")
        return

    finite_df = df.loc[finite_mask].copy()

    # ------------------------------------------------------------
    # Sort by energy
    # ------------------------------------------------------------

    sorted_df = finite_df.sort_values(
        by="center_energy",
        ascending=True
    ).reset_index(drop=True)

    print()
    print("Energy statistics")
    print("-" * 60)

    print(f"Minimum energy:  {sorted_df['center_energy'].iloc[0]:.15e}")
    print(f"Maximum energy:  {sorted_df['center_energy'].iloc[-1]:.15e}")
    print(f"Mean energy:     {sorted_df['center_energy'].mean():.15e}")
    print(f"Median energy:   {sorted_df['center_energy'].median():.15e}")
    print(f"Std energy:      {sorted_df['center_energy'].std():.15e}")

    # ------------------------------------------------------------
    # Gradient statistics
    # ------------------------------------------------------------

    grad = sorted_df["gradient_norm"].to_numpy()
    finite_grad = grad[np.isfinite(grad)]

    if len(finite_grad) > 0:
        print()
        print("Gradient norm statistics")
        print("-" * 60)

        print(f"Minimum gradient norm: {np.min(finite_grad):.15e}")
        print(f"Maximum gradient norm: {np.max(finite_grad):.15e}")
        print(f"Mean gradient norm:    {np.mean(finite_grad):.15e}")
        print(f"Median gradient norm:  {np.median(finite_grad):.15e}")
        print(f"Std gradient norm:     {np.std(finite_grad):.15e}")

    # ------------------------------------------------------------
    # Print lowest-energy points
    # ------------------------------------------------------------

    print()
    print(f"Lowest {min(args.top, len(sorted_df))} finite-energy points")
    print("-" * 60)

    cols = [
        c for c in [
            "depth",
            "cell_index",
            "center_energy",
            "gradient_norm",
            "volume",
        ]
        if c in sorted_df.columns
    ]

    print(
        sorted_df[cols]
        .head(args.top)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Save sorted file
    # ------------------------------------------------------------

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_sorted_by_energy.csv"

    sorted_df.to_csv(args.output, index=False)

    print()
    print(f"Wrote sorted finite-energy data to:")
    print(f"  {args.output}")

    # ------------------------------------------------------------
    # Plot 1: Energy distribution
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.hist(
        sorted_df["center_energy"],
        bins=args.bins
    )

    plt.xlabel("Center energy")
    plt.ylabel("Number of cells")
    plt.title("Distribution of Cell Center Energies")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    energy_plot = os.path.splitext(args.output)[0] + "_energy_distribution.png"
    plt.savefig(energy_plot, dpi=200)

    print(f"Wrote energy distribution:")
    print(f"  {energy_plot}")

    # ------------------------------------------------------------
    # Plot 2: Gradient norm distribution
    # ------------------------------------------------------------

    if len(finite_grad) > 0:

        plt.figure(figsize=(9, 6))

        plt.hist(
            finite_grad,
            bins=args.bins
        )

        plt.xlabel(r"$\|\nabla E\|$")
        plt.ylabel("Number of cells")
        plt.title("Distribution of Gradient Norms")
        plt.grid(alpha=0.25)

        plt.tight_layout()

        gradient_plot = (
            os.path.splitext(args.output)[0]
            + "_gradient_distribution.png"
        )

        plt.savefig(gradient_plot, dpi=200)

        print(f"Wrote gradient distribution:")
        print(f"  {gradient_plot}")

    # ------------------------------------------------------------
    # Plot 3: Energy versus sorted rank
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    ranks = np.arange(len(sorted_df))

    plt.plot(
        ranks,
        sorted_df["center_energy"].to_numpy(),
        linewidth=1
    )

    plt.xlabel("Point rank (sorted by energy)")
    plt.ylabel("Center energy")
    plt.title("Sorted Cell Energies")
    plt.grid(alpha=0.25)

    plt.tight_layout()

    rank_plot = (
        os.path.splitext(args.output)[0]
        + "_energy_sorted.png"
    )

    plt.savefig(rank_plot, dpi=200)

    print(f"Wrote sorted-energy plot:")
    print(f"  {rank_plot}")

    # ------------------------------------------------------------
    # Plot 4: Gradient norm versus energy
    # ------------------------------------------------------------

    if len(finite_grad) == len(sorted_df):

        plt.figure(figsize=(9, 6))

        plt.scatter(
            sorted_df["center_energy"],
            sorted_df["gradient_norm"],
            s=4
        )

        plt.xlabel("Center energy")
        plt.ylabel(r"$\|\nabla E\|$")
        plt.title("Gradient Norm vs. Center Energy")
        plt.grid(alpha=0.25)

        plt.tight_layout()

        scatter_plot = (
            os.path.splitext(args.output)[0]
            + "_energy_vs_gradient.png"
        )

        plt.savefig(scatter_plot, dpi=200)

        print(f"Wrote energy/gradient plot:")
        print(f"  {scatter_plot}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_parameter_mesh(
        cells,
        particle_indexes=[0,1,2],
        lower_bounds=None,
        show_centers=True,
        cmap="viridis"):
    """
    Plot the search mesh for one particle in (phi, theta) space.

    Parameters
    ----------
    cells : list[Cell]

    particle_indexes : list[int]
        Which particle indexes to visualize.

    lower_bounds : list or None
        Optional value used to color each cell.

    show_centers : bool
        Plot midpoint of every cell.

    """

    fig, ax = plt.subplots(figsize=(10,5))

    if lower_bounds is None:
        lower_bounds = np.arange(len(cells))

    norm = plt.Normalize(
        min(lower_bounds),
        max(lower_bounds)
    )

    cm = plt.cm.get_cmap(cmap)

    for cell, value in zip(cells, lower_bounds):

        for particle_index in particle_indexes:
            particle = cell.particle_ranges[particle_index]
            theta_bounds = particle.bounds[0]
            phi_bounds = particle.bounds[1]

            t0, t1 = theta_bounds.lo, theta_bounds.hi
            p0, p1 = phi_bounds.lo, phi_bounds.hi

            if show_centers:

                ax.plot(
                    (t0+t1)/2,
                    (p0+p1)/2,
                    '.k',
                    markersize=2
                )
            else:
                rect = Rectangle(
                   (t0, p0),
                   t1-t0,
                   p1-p0,
                   facecolor=cm(norm(value)),
                   edgecolor='k',
                   linewidth=0.25,
                   alpha=0.5
                )

                ax.add_patch(rect)



    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,np.pi)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')

    ax.set_xticks([
        0,
        np.pi/2,
        np.pi,
        3*np.pi/2,
        2*np.pi
    ])

    ax.set_xticklabels([
        "0",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
        r"$2\pi$"
    ])

    ax.set_yticks([
        0,
        np.pi/2,
        np.pi
    ])

    ax.set_yticklabels([
        "0",
        r"$\pi/2$",
        r"$\pi$"
    ])

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cm),
        ax=ax,
        label="Lower bound"
    )

    plt.tight_layout()
    plt.show()
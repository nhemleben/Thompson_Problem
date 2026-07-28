import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_parameter_mesh(
        cells,
        particles=[0,1,2],
        lower_bounds=None,
        show_centers=True,
        cmap="viridis"):
    """
    Plot the search mesh for one particle in (phi, theta) space.

    Parameters
    ----------
    cells : list[Cell]

    particle : int
        Which particle to visualize.

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




#Make this as subplots or different plots or remove rectable
#more importantly fix search so the base one is at 0.0 in polar and so that the 
#second particle is on the equator (theta = pi/2)
#




        for particle in particles:
            theta_bounds, phi_bounds = cell.bounds[particle]

            t0, t1 = theta_bounds
            p0, p1 = phi_bounds

            rect = Rectangle(
                (p0, t0),
                p1-p0,
                t1-t0,
                facecolor=cm(norm(value)),
                edgecolor='k',
                linewidth=0.25,
                alpha=0.5
            )

            ax.add_patch(rect)

            if show_centers:

                ax.plot(
                    (p0+p1)/2,
                    (t0+t1)/2,
                    '.k',
                    markersize=2
                )

    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,np.pi)

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')

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
import numpy as np
import matplotlib.pyplot as plt

from geometry import spherical_to_cart


def draw_global_search(
        cells,
        lower_bounds=None,
        particle=0):
    """
    Plot active search cells globally.

    cells:
        list of Cell objects

    lower_bounds:
        list of energy lower bounds

    particle:
        which particle coordinate to display
    """

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # sphere

    u=np.linspace(0,2*np.pi,80)
    v=np.linspace(0,np.pi,40)

    X=np.outer(
        np.cos(u),
        np.sin(v)
    )

    Y=np.outer(
        np.sin(u),
        np.sin(v)
    )

    Z=np.outer(
        np.ones_like(u),
        np.cos(v)
    )

    ax.plot_wireframe(
        X,Y,Z,
        linewidth=0.25,
        alpha=0.2
    )


    points=[]


    for cell in cells:

        theta_bounds,phi_bounds = (
            cell.bounds[particle]
        )


        theta=np.mean(theta_bounds)
        phi=np.mean(phi_bounds)


        p=spherical_to_cart(
            theta,
            phi
        )

        points.append(p)


    points=np.array(points)


    if lower_bounds is None:

        colors=np.arange(
            len(points)
        )

    else:

        colors=np.array(
            lower_bounds
        )


    scatter=ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        c=colors,
        s=30
    )


    fig.colorbar(
        scatter,
        label="Energy lower bound"
    )


    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


    plt.show()
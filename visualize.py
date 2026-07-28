import numpy as np
import matplotlib.pyplot as plt

from geometry import spherical_to_cart


def draw_sphere(ax):

    u=np.linspace(0,2*np.pi,60)
    v=np.linspace(0,np.pi,30)

    x=np.outer(np.cos(u),np.sin(v))
    y=np.outer(np.sin(u),np.sin(v))
    z=np.outer(np.ones_like(u),np.cos(v))

    ax.plot_wireframe(
        x,y,z,
        linewidth=0.3,
        alpha=0.2
    )



def draw_cell(ax, bounds, color="r"):

    """
    Draw spherical rectangle on S^2

    bounds:
        [
          [theta_min,theta_max],
          [phi_min,phi_max]
        ]
    """

    theta_bounds,phi_bounds=bounds

    theta=np.linspace(
        theta_bounds[0],
        theta_bounds[1],
        20
    )

    phi=np.linspace(
        phi_bounds[0],
        phi_bounds[1],
        20
    )


    T,P=np.meshgrid(
        theta,
        phi
    )


    X=np.zeros_like(T)
    Y=np.zeros_like(T)
    Z=np.zeros_like(T)


    for i in range(T.shape[0]):
        for j in range(T.shape[1]):

            p=spherical_to_cart(
                T[i,j],
                P[i,j]
            )

            X[i,j]=p[0]
            Y[i,j]=p[1]
            Z[i,j]=p[2]


    ax.plot_surface(
        X,Y,Z,
        alpha=0.35
    )



def draw_center(ax,bounds):

    theta=(
        bounds[0][0]+bounds[0][1]
    )/2

    phi=(
        bounds[1][0]+bounds[1][1]
    )/2


    p=spherical_to_cart(
        theta,
        phi
    )

    ax.scatter(
        p[0],
        p[1],
        p[2],
        s=50
    )



def visualize_cell(cell):

    n=len(cell.bounds)


    fig=plt.figure(
        figsize=(5*n,5)
    )


    for i,bounds in enumerate(cell.bounds):

        ax=fig.add_subplot(
            1,n,i+1,
            projection="3d"
        )

        ax.set_title(
            f"Particle {i+1}"
        )

        draw_sphere(ax)

        draw_cell(
            ax,
            bounds
        )

        draw_center(
            ax,
            bounds
        )


        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])


    plt.show()
from partition import Particle_Ranges, Bounds, Cell
import numpy as np
from bound import angle_min


def initial_cell_non_antipodal(n):
    """
    Construct the initial cell for the non-antipodal configuration of n particles on a sphere.
    We avoid the 'south' pole by phi_min/2 to ensure that we do not have a antipolar configuration.

    Parameters:
    n (int): Number of particles.

    Returns:
    Cell: The initial cell with specified bounds for each particle.
    """

    theta_easment = 1.0/10.0  # easment for the theta coordinate chart of third particle only
        #Know that the n=3 solution has the third particle near the equator, so we allow a small easment
    phi_min = angle_min(n)
    phi_restriction = phi_min/2  # restriction for the phi coordinate chart

    bounds = [
        # particle 1 (fixed) at pole
        Particle_Ranges(
            bounds = [ Bounds(0.0,0.0), Bounds(0.0,0.0) ],
            fixed = [True, True]
        ),
        # particle 2 fixed on equator that goes through pole
        Particle_Ranges(
            bounds = [ Bounds(0.0, 0.0), Bounds(phi_min,np.pi - phi_restriction) ],
            fixed = [True, False]
        )
    ]

    if n >= 3:
        bounds.append(
            # particle 3 is limit to just 'eastern' hemisphere due to reflection symetry
            Particle_Ranges(
                bounds = [ Bounds(0.0,np.pi +theta_easment), Bounds(phi_min,np.pi - phi_restriction,) ], 
                fixed = [False, False]
            )
        )

    for i in range(n-3):
        bounds.append(
            Particle_Ranges(
                bounds = [ Bounds(0.0,2*np.pi,), Bounds(phi_min,np.pi - phi_restriction,) ], 
                fixed = [False, False]
            )
        )

    return Cell(bounds)



def initial_cell_antipodal(n):
    """
    Construct the initial cell for the antipodal configuration of n particles on a sphere.

    Parameters:
    n (int): Number of particles.

    Returns:
    Cell: The initial cell with specified bounds for each particle.
    """
    theta_easment = 1/10  # easment for the theta coordinate chart of third particle only
    phi_easment = 1/10  # easment for the phi coordinate chart of the last particle at the south pole
        #Know that the n=3 solution has the third particle near the equator, so we allow a small easment
    phi_min = angle_min(n)
    phi_restriction = phi_min/2  # restriction for the phi coordinate chart

    bounds = [
        # particle 1 (fixed) at pole
        Particle_Ranges(
            bounds = [ Bounds(0.0,0.0), Bounds(0.0,0.0) ],
            fixed = [True, True]
        ),
        # particle 2 fixed on equator that goes through pole
        Particle_Ranges(
            bounds = [ Bounds(0.0, 0.0), Bounds(phi_min,np.pi - phi_restriction/2) ],
            fixed = [True, False]
        )
    ]

    if n >= 3:
        if n >3:
            bounds.append(
                # particle 3 is limit to just 'eastern' hemisphere due to reflection symetry
                # Third particle when it is not the last and therefore polar particle
                Particle_Ranges(
                    bounds = [ Bounds(0.0,np.pi +theta_easment), Bounds(phi_min,np.pi - phi_restriction/2) ], 
                    fixed = [False, False]
                )
            )

            for i in range(n-4):
                bounds.append(
                    Particle_Ranges(
                        bounds = [ Bounds(0.0,2*np.pi,), Bounds(phi_min,np.pi - phi_restriction/2) ], 
                        fixed = [False, False]
                    )
                )

        #Last particle lives on the south pole
        bounds.append(
            Particle_Ranges(
                bounds = [ Bounds(0.0,2*np.pi,), Bounds(0,phi_restriction + phi_easment) ], 
                fixed = [False, False]
            )
        )

    return Cell(bounds)


def initial_cell(n, antipodal=False, mode=None):
    """
    Backward-compatible initial-cell selector.

    Parameters
    ----------
    n : int
        Number of particles.
    antipodal : bool
        When True, use antipodal initialization.
    mode : str | None
        Optional explicit mode: "antipodal" or "non-antipodal".
        If provided, it takes precedence over `antipodal`.
    """

    selected_mode = mode
    if selected_mode is None:
        selected_mode = "antipodal" if antipodal else "non-antipodal"

    normalized = str(selected_mode).strip().lower().replace("_", "-")

    if normalized == "antipodal":
        return initial_cell_antipodal(n)

    if normalized in {"non-antipodal", "nonantipodal"}:
        return initial_cell_non_antipodal(n)

    raise ValueError(
        f"Unknown initial-cell mode '{selected_mode}'. "
        "Expected 'antipodal' or 'non-antipodal'."
    )



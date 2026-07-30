from partition import Particle_Ranges, Bounds, Cell
import numpy as np
from bound import angle_min


def initial_cell(n):
    phi_restriction = 0.1  # restriction for the phi coordinate chart
    phi_min = angle_min(n)

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
                bounds = [ Bounds(0.0,np.pi,), Bounds(phi_min,np.pi - phi_restriction,) ], 
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





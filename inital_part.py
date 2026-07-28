from partition import Particle_Ranges, Bounds, Cell
import numpy as np


def initial_cell(n):

    bounds = [
        # particle 1 (fixed) at pole
        Particle_Ranges(
            bounds = [ Bounds(0.0,0.0), Bounds(0.0,0.0) ],
            fixed = [True, True]
        ),
        # particle 2 fixed on equator that goes through pole
        Particle_Ranges(
            bounds = [ Bounds(0.0, 0.0), Bounds(0.0,np.pi) ],
            fixed = [True, False]
        )
    ]

#TODO: smaller window on initial particles since these can not be too close to the first 2 particles
    for i in range(n-2):
        bounds.append(
            Particle_Ranges(
                bounds = [ Bounds(0.0,2*np.pi,), Bounds(0.0,np.pi,) ], 
                fixed = [False, False]
            )
        )

    return Cell(bounds)
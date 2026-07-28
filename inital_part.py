from partition import Cell
import numpy as np


def initial_cell(n):

    bounds = [
    # particle 1 (fixed) at pole
    [
        [0.0, 0.0],
        [0.0, 0.0]
    ],

    # particle 2 fixed on equator that goes through pole
    [
        [0.0, np.pi],
        [0.0, 0.0]
    ],

]

    for i in range(n-2):
        bounds.append(
            [
                [0.0, np.pi], #TO DO: There should be better theta/phi upper/lower bounds just based on 
                [0.0, 2*np.pi] #Energy arguments here
            ]
        )

    return Cell(bounds)
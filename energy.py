import itertools
import numpy as np
from geometry import spherical_to_cart


def thompson_energy(points):

    """
    Coulomb energy on sphere
    """

    xyz=[
        spherical_to_cart(*p)
        for p in points
    ]

    E=0

    for i,j in itertools.combinations(
            range(len(xyz)),2):

        d=np.linalg.norm(
            xyz[i]-xyz[j]
        )

        E += 1/d

    return E
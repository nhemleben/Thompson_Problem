import numpy as np
from geometry import spherical_to_cart


def corner_points(box):

    theta=[ box[0].lo, box[0].hi ]
    phi=[ box[1].lo, box[1].hi]

    pts=[]

    for t in theta:

        for p in phi:

            pts.append(
                spherical_to_cart(t,p)
            )

    return pts



def pair_lower_bound(box1,box2):

    """ Minimum possible Coulomb contribution """

    c1=corner_points(box1)
    c2=corner_points(box2)

    maxd=0

    for a in c1:
        for b in c2:

            d=np.linalg.norm(a-b)

            maxd=max(maxd,d)

    if maxd == 0:
        return float("inf")

    return 1/maxd



def energy_lower_bound(cell):

    E=0

    bounds =[pr.bounds for pr in cell.particle_ranges]

    n=len(bounds)

    for i in range(n):
        for j in range(i+1,n):

            E += pair_lower_bound( bounds[i], bounds[j])

    return E
import numpy as np
from intvalpy import Interval

def interval_box(q_star, h):
    """
    Construct an interval box centered at q_star.

    Parameters
    ----------
    q_star : ndarray
        Candidate minimizer.
    h : float
        Half side length.

    Returns
    -------
    list
        List of Interval objects.
    """

    return [Interval(x - h, x + h) for x in q_star]



def gershgorin_pd(H):
    """
    Rigorous positive definiteness test using Gershgorin.

    Parameters
    ----------
    H : list[list[Interval]]
        Interval matrix (Hessian) to be tested for positive definiteness.

    Returns
    -------
    bool
    """

    n = len(H)

    for i in range(n):

        #
        # Lower bound of diagonal interval
        #
        diag_lower = H[i][i].a

        #
        # Upper bound of Gershgorin radius
        #
        radius = 0.0

        for j in range(n):

            if i == j:
                continue

            hij = H[i][j]

            radius += max(abs(hij.a), abs(hij.b))

        if diag_lower <= radius:
            return False

    return True

def verify_box(q_star, h):

    qI = interval_box(q_star, h)

    H = spherical_hessian(qI)

    return gershgorin_pd(H)



def maximal_convex_box(q_star,
                       h0=1e-6,
                       growth=2.0,
                       tol=1e-12,
                       max_iter=60):

    #
    # Expand until failure
    #

    h_low = 0.0
    h_high = h0

    while verify_box(q_star, h_high):
        h_low = h_high
        h_high *= growth

    #
    # Binary search
    #

    for _ in range(max_iter):

        h = 0.5 * (h_low + h_high)

        if verify_box(q_star, h):
            h_low = h
        else:
            h_high = h

        if h_high - h_low < tol:
            break

    return h_low
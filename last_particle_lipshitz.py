import numpy as np


def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


def torque_vector(x, points):
    """
    T(x) = sum_j (p_j x x)/|x-p_j|^3
    """

    T = np.zeros(3)

    for p in points:
        d = np.linalg.norm(x - p)
        T += np.cross(p, x) / d**3

    return T


def psi_value(x, points):
    """
    Psi(x) = |T(x)|^2
    """
    T = torque_vector(x, points)
    return np.dot(T, T)


def torque_lipschitz(x, points, cell_radius):
    """
    Upper bound on ||grad T|| over the cell.

    Uses

        ||grad T_j||
            <= 1/d^3 + 3/d^4
    """

    LT = 0.0

    for p in points:

        d = np.linalg.norm(x - p)

        # Lower bound on distance everywhere in the cell
        delta = max(d - cell_radius, 1e-12)

        #Adding cross for tighter bound here
        cross = np.cross(p, x)
        cross_norm = np.linalg.norm(cross)

        LT += (delta**(-3)) + cross_norm*3.0 * delta**(-4)

    return LT


def psi_lipschitz(x, points, cell_radius):
    """
    Improved Lipschitz constant

        L = 2 (||T(x)|| + LT*r) LT
    """

    T = torque_vector(x, points)

    Tnorm = np.linalg.norm(T)

    LT = torque_lipschitz(x, points, cell_radius)

    return 2.0 * (Tnorm + LT * cell_radius) * LT

def cross_matrix(p):
    """
    Returns the skew-symmetric matrix C such that

        C @ v = p x v
    """
    px, py, pz = p

    return np.array([
        [0.0, -pz,  py],
        [pz,   0.0, -px],
        [-py,  px,   0.0]
    ])


def torque_jacobian(x, points):
    """
    Exact Jacobian

        J = dT/dx

    J = sum [
            C(p)/d^3
          - 3 (p x x) r^T / d^5
        ]
    """

    J = np.zeros((3, 3))

    for p in points:

        r = x - p
        d = np.linalg.norm(r)

        cross = np.cross(p, x)

        J += (
            cross_matrix(p) / d**3
            -
            3.0 * np.outer(cross, r) / d**5
        )

    return J

def spherical_jacobian(J, x):
    """
    Restrict a 3x3 Jacobian to the tangent plane of the unit sphere.

    Parameters
    ----------
    J : (3,3) ndarray
        Ambient Cartesian Jacobian.

    x : (3,) ndarray
        Point on the unit sphere.

    Returns
    -------
    Js : (3,2) ndarray
        Spherical (tangent) Jacobian.
    """

    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    # Pick a vector not parallel to x
    if abs(x[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])

    # Tangent basis
    e1 = np.cross(a, x)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(x, e1)

    # Restrict J to tangent directions
    E = np.column_stack((e1, e2))

    return J @ E

def exclusion_test(theta,
                   phi,
                   points,
                   cell_radius):
    """
    Returns whether a spherical mesh cell can be discarded.
    """

    x = spherical_to_cartesian(theta, phi)

    psi = psi_value(x, points)

    L = psi_lipschitz(x, points, cell_radius)

    removable = psi > L * cell_radius

    return {
        "psi": psi,
        "Lpsi": L,
        "bound": L * cell_radius,
        "discard": removable
    }


###########################################################
# Example
###########################################################

if __name__ == "__main__":

    points = np.array([
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.]
    ])

    theta = 1.2
    phi = 2.0

    cell_radius = 0.02

    result = exclusion_test(
        theta,
        phi,
        points,
        cell_radius
    )

    print(result)
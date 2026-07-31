import math
import numpy as np

def T_n(n):
    """
    Return known optimal Thomson problem energies T_n.

    Values are stored for cases where exact/numerical optima are known.
    Extend this dictionary as needed.
    """
    T = {
        1: 0, #this is trivial, only one particle
        2: 0.5,
        3: 1.732050808,
        4: 3.674234614,
        5: 6.474691495,
        6: 9.985281374,
        7: 14.452977415,
        8: 19.675287861,
        9: 25.759986531,
        10:32.716949460,
        11:40.596450510,
    }

    if n not in T:
        raise ValueError(f"No known T_n stored for n={n}")

    return T[n]

def L_n(n):
    return T_n(n) - T_n(n-1)




def Lower_n(n):
    """
    This is not my convention but kept just to see
    Return the asymptotic lower estimate L_n.

    Uses the leading-order Thomson energy estimate:
        E_n ~ n^2/2 - alpha*n^(3/2)

    Replace alpha if using a tighter published bound.
    """
    alpha = 1.106103  # hexagonal lattice constant approximation

    return 0.5*n*n - alpha*n**1.5


def cartesian_to_spherical(cartesian):
    x, y, z = cartesian
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2*math.pi

    phi = math.acos(z / r)

    return theta, phi

def spherical_configuration(points):
    return [cartesian_to_spherical(p) for p in points]


KNOWN_SOLUTIONS = {
      2: np.array([
        [0,0,1],
        [0,0,-1],
    ], dtype=float),

    3: np.array([
        [0,0,1],
        [ math.sqrt(3)/2, 0,-0.5],
        [-math.sqrt(3)/2, 0,-0.5],
    ], dtype=float),

    4: np.array([
        [ 0, 0, 1],
        [ 2 * np.sqrt(2)/3 ,0,-1/3],
        [- np.sqrt(2)/3, np.sqrt(6)/3,-1/3],
        [-np.sqrt(2)/3, -np.sqrt(6)/3,-1/3],
    ], dtype=float),

    5: np.array([
        [0,0,1],
        [ math.sqrt(3)/2,0,-0.5,],
        [-math.sqrt(3)/2,0,-0.5,],
        [0,1,0],
        [0,-1,0],
    ], dtype=float),


#TODO: This will most likely need the special antipodal chart
    6: np.array([
        [0,0, 1],
        [0,0,-1],
        [ 1,0,0],
        [-1,0,0],
        [0, 1,0],
        [0,-1,0],
    ], dtype=float),


    7: np.array([
        [0,0,1],
        [0,0,-1],
        [1,0,0],
        [ math.cos(2*math.pi/5),  math.sin(2*math.pi/5), 0],
        [ math.cos(4*math.pi/5),  math.sin(4*math.pi/5), 0],
        [ math.cos(6*math.pi/5),  math.sin(6*math.pi/5), 0],
        [ math.cos(8*math.pi/5),  math.sin(8*math.pi/5), 0],
    ], dtype=float),


}
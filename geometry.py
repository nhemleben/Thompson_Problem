import numpy as np


def spherical_to_cart(theta, phi, chart="standard"):
    """
    Map spherical coordinates to S^2
    """
    if chart == "antipodal_psi":
        psi = phi
        return np.array([
            np.sin(psi) * np.cos(theta),
            np.sin(psi) * np.sin(theta),
            -np.cos(psi),
        ])

    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def chord_distance(a,b):
    return np.linalg.norm(a-b)


def angle_distance(x1,x2):

    theta1,phi1=x1
    theta2,phi2=x2

    c = (
        np.cos(phi1)*np.cos(phi2)
        +
        np.sin(phi1)*np.sin(phi2)
        *
        np.cos(theta1-theta2)
    )

    c=np.clip(c,-1,1)

    return np.arccos(c)
import numpy as np


def spherical_to_cart(theta, phi):
    """
    Map spherical coordinates to S^2
    """
    return np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])


def chord_distance(a,b):
    return np.linalg.norm(a-b)


def angle_distance(x1,x2):

    theta1,phi1=x1
    theta2,phi2=x2

    c = (
        np.cos(theta1)*np.cos(theta2)
        +
        np.sin(theta1)*np.sin(theta2)
        *
        np.cos(phi1-phi2)
    )

    c=np.clip(c,-1,1)

    return np.arccos(c)
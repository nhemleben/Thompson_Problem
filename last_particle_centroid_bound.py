import numpy as np
from energy import thompson_energy_last_particle


def cartesian_to_spherical(p):
    """
    Returns (theta, phi)

    theta = azimuth in [0,2pi)
    phi   = polar angle from +z in [0,pi]
    """
    x, y, z = p
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2*np.pi

    phi = np.arccos(np.clip(z, -1.0, 1.0))

    return theta, phi


def last_particle_cap(points, U_n=None, E_prev = None):
    """
    points : (n,3) numpy array
    U_n    : upper bound on Thomson energy for n particles
    E_prev : energy of the first n-1 particles

    Returns:
        center_theta
        center_phi
        angular_radius
    """
    if U_n is None or E_prev is None:
        #You only need the nt particles contribution here
        nth_particle_energy = thompson_energy_last_particle(points)
    else:
        nth_particle_energy = U_n - E_prev
    n = len(points) 
    S = np.sum(points, axis=0)
    S_norm = np.linalg.norm(S)


#Original way to write, idk why it was written this way tbh
#    Cmax2 = (
#        S_norm**2 + 1 + (n-1) * ( 2 - ((n-1)/(nth_particle_energy)**2)
#        )
#    ) 
#    b = (Cmax2 - S_norm**2 - 1)/2

    b =  (n-1) * ( 1 - (1/2) * ((n-1)/(nth_particle_energy)**2))

    if S_norm < 1e-12:
        return {
            "all_sphere": True,
            "center_theta": None,
            "center_phi": None,
            "angular_radius": np.pi
        }

    c = b / S_norm

    if c <= -1:
        return {
            "empty": True
        }

    if c >= 1:
        return {
            "all_sphere": True,
            "center_theta": None,
            "center_phi": None,
            "angular_radius": np.pi
        }

    alpha = np.arccos(c)

    center = -S / S_norm

    theta, phi = cartesian_to_spherical(center)

    return {
        "center_theta": theta,
        "center_phi": phi,
        "angular_radius": alpha,
        "center_vector": center,
        "S_norm": S_norm,
        "dot_bound": b,
    }


def last_particle_cap_tighter(points, U_n=None, E_prev = None):
    """
    points : (n,3) numpy array
    U_n    : upper bound on Thomson energy for n particles
    E_prev : energy of the first n-1 particles

    Returns:
        center_theta
        center_phi
        angular_radius
    """
    if U_n is None or E_prev is None:
        #You only need the nt particles contribution here
        nth_particle_energy = thompson_energy_last_particle(points)
    else:
        nth_particle_energy = U_n - E_prev
    n = len(points) 
    S = np.sum(points, axis=0)
    S_norm = np.linalg.norm(S)

    b_original =  (n-1) * ( 1 - (1/2) * ((n-1)/(nth_particle_energy)**2))

    all_n_distances =[]
    for i in range(n-1):
            dist = np.linalg.norm(points[i] - points[n-1])
            all_n_distances.append(dist)


    D_1 = sum(all_n_distances)
    D_2 = sum([d**2 for d in all_n_distances])

    Del_1 = D_1 * nth_particle_energy - (n-1)**2
    Del_2 = D_2 * nth_particle_energy - D_1**2

    correction_factors =  (n-1) * Del_1 / nth_particle_energy**2 + (
             Del_1**2 / (2*(n-1) * nth_particle_energy**2) +
             Del_2 / (2 * (n-1))
    )

    b = b_original - (correction_factors)

    if S_norm < 1e-12:
        return {
            "all_sphere": True,
            "center_theta": None,
            "center_phi": None,
            "angular_radius": np.pi
        }

    c = b / S_norm

    if c <= -1:
        return {
            "empty": True
        }

    if c >= 1:
        return {
            "all_sphere": True,
            "center_theta": None,
            "center_phi": None,
            "angular_radius": np.pi
        }

    alpha = np.arccos(c)

    center = -S / S_norm

    theta, phi = cartesian_to_spherical(center)

    return {
        "center_theta": theta,
        "center_phi": phi,
        "angular_radius": alpha,
        "center_vector": center,
        "S_norm": S_norm,
        "dot_bound": b,
    }





def cap_to_theta_phi_bounds(center_theta, center_phi, alpha):
    """
    Conservative rectangular bounds containing the spherical cap.
    """

    phi_min = max(0.0, center_phi - alpha)
    phi_max = min(np.pi, center_phi + alpha)

    if np.sin(center_phi) < 1e-12:
        # cap touches a pole
        theta_min = 0.0
        theta_max = 2*np.pi
    else:
        denom = np.sin(center_phi)
        t = np.sin(alpha) / denom

        if t >= 1:
            theta_min = 0.0
            theta_max = 2*np.pi
        else:
            dtheta = np.arcsin(t)
            theta_min = center_theta - dtheta
            theta_max = center_theta + dtheta

    return theta_min, theta_max, phi_min, phi_max
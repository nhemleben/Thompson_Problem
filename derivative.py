import numpy as np


def thomson_energy(X):
    """
    Coulomb energy for Thomson problem.

    Parameters
    ----------
    X : ndarray
        Shape (N,3), particle coordinates on sphere.

    Returns
    -------
    float
        Energy
    """
    N = len(X)
    E = 0.0

    for i in range(N):
        for j in range(i+1, N):
            r = X[i] - X[j]
            E += 1.0 / np.linalg.norm(r)

    return E


def thomson_gradient(X):
    """
    Gradient of Thomson energy.

    Returns
    -------
    grad : ndarray
        Shape (N,3)
    """

    N = len(X)
    grad = np.zeros_like(X)

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            d = X[i] - X[j]
            r = np.linalg.norm(d)

            # d(1/r)/dx_i = -d/r^3
            grad[i] -= d / r**3

    return grad


def thomson_hessian(X):
    """
    Hessian of Thomson energy.

    Returns
    -------
    H : ndarray
        Shape (3N,3N)

    """

    N = len(X)
    H = np.zeros((3*N,3*N))

    for i in range(N):
        for j in range(i+1,N):

            d = X[i]-X[j]
            r = np.linalg.norm(d)

            I = np.eye(3)

            # Hessian block:
            # (3 dd^T - r^2 I)/r^5
            block = (
                3*np.outer(d,d) - r*r*I
            ) / r**5


            # diagonal blocks
            H[3*i:3*i+3,3*i:3*i+3] += block
            H[3*j:3*j+3,3*j:3*j+3] += block

            # off diagonal blocks
            H[3*i:3*i+3,3*j:3*j+3] -= block
            H[3*j:3*j+3,3*i:3*i+3] -= block


    return H





def spherical_jacobian(theta, phi, chart="standard"):
    """
    Jacobian dx/d(theta,phi)

    Returns:
        J : (3,2)
    """

    s = np.sin(phi)
    c = np.cos(phi)

    ct = np.cos(theta)
    st = np.sin(theta)


    # x = sin(phi) cos(theta)
    dtheta = np.array([
        -s*st,
        s*ct,
        0
    ])

    if chart == "antipodal_psi":
        dphi = np.array([
            c * ct,
            c * st,
            s,
        ])
    else:
        # d/dphi
        dphi = np.array([
            c*ct,
            c*st,
            -s
        ])

    return np.column_stack((dtheta,dphi))



def spherical_hessian_from_cartesian(
        X,
        H_cart,
    grad_cart=None,
    charts=None):
    """
    Convert Cartesian Hessian to spherical Hessian.

    Parameters
    ----------
    X:
        (N,3) particle coordinates

    H_cart:
        (3N,3N) Cartesian Hessian

    grad_cart:
        (3N,) Cartesian gradient

    Returns
    -------
    H_sph:
        (2N,2N)
    """

    N = len(X)


    # ----------------------------
    # Construct block Jacobian
    # ----------------------------

    J = np.zeros((3*N,2*N))


    if charts is None:
        charts = ["standard"] * N
    elif len(charts) != N:
        raise ValueError("charts length must match number of particles")

    angles=[]

    for i,p in enumerate(X):

        x,y,z=p

        theta=np.arctan2(y,x)

        chart = charts[i]
        if chart == "antipodal_psi":
            phi=np.arccos(-z)
        else:
            phi=np.arccos(z)

        angles.append((theta,phi,chart))


        J[
            3*i:3*i+3,
            2*i:2*i+2
        ] = spherical_jacobian(
            theta,
            phi,
            chart=chart,
        )


    # Main Hessian transform
    Hs = J.T @ H_cart @ J


    # ------------------------------------------------
    # Optional curvature correction term
    # ------------------------------------------------

    if grad_cart is not None:

        for i,(theta,phi,chart) in enumerate(angles):

            s=np.sin(phi)
            c=np.cos(phi)

            ct=np.cos(theta)
            st=np.sin(theta)


            # second derivatives of position

            dtt=np.array([
                -s*ct,
                -s*st,
                0
            ])

            dtp=np.array([
                -c*st,
                c*ct,
                0
            ])

            if chart == "antipodal_psi":
                dpp=np.array([
                    -s*ct,
                    -s*st,
                    c
                ])
            else:
                dpp=np.array([
                    -s*ct,
                    -s*st,
                    -c
                ])


            second=[
                dtt,dtp,dpp
            ]


            for a in range(2):

                for b in range(2):

                    value=0

                    if a==0 and b==0:
                        vec=dtt

                    elif a==1 and b==1:
                        vec=dpp

                    else:
                        vec=dtp


                    value = (
                        grad_cart[3*i:3*i+3]
                        @ vec
                    )


                    Hs[
                        2*i+a,
                        2*i+b
                    ] += value


    return Hs
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
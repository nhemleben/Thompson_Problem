import sympy as sp


def generate_thomson_derivatives(N):
    """
    Generates symbolic energy, gradient, Hessian and optimized
    Python code for the Thomson problem on the sphere.

    Coordinates:
        theta_i  in [0,2pi)
        phi_i    in [0,pi]

    Returns
    -------
    energy
    gradient
    hessian
    """

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    theta = sp.symbols(f'theta0:{N}', real=True)
    phi = sp.symbols(f'phi0:{N}', real=True)

    # Variable ordering:
    variables = []

    for i in range(N):
        variables.append(theta[i])
        variables.append(phi[i])

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    E = 0

    for i in range(N):

        for j in range(i+1, N):

            c = (
                sp.cos(phi[i])*sp.cos(phi[j])
                + sp.sin(phi[i])*sp.sin(phi[j])
                * sp.cos(theta[i]-theta[j])
            )

            r = sp.sqrt(2-2*c)

            E += 1/r

    E = sp.simplify(E)

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    print("Computing gradient...")

    gradient = [

        sp.simplify(
            sp.diff(E,v)
        )

        for v in variables
    ]

    # --------------------------------------------------------
    # Hessian
    # --------------------------------------------------------

    print("Computing Hessian...")

    H = sp.Matrix([gradient]).jacobian(variables)

    H = H.applyfunc(sp.simplify)

    return E, variables, gradient, H


######################################################################
# Code generation
######################################################################


def generate_python_code(N):

    E, vars, grad, H = generate_thomson_derivatives(N)

    print("Running common subexpression elimination...")

    expressions = [E] + grad + list(H)

    replacements, reduced = sp.cse(
        expressions,
        optimizations='basic'
    )

    print("\n##################################################")
    print("# Common Subexpressions")
    print("##################################################\n")

    for lhs, rhs in replacements:
        print(f"{lhs} = {sp.pycode(rhs)}")

    print("\n##################################################")
    print("# Energy")
    print("##################################################\n")

    print(f"E = {sp.pycode(reduced[0])}")

    print("\n##################################################")
    print("# Gradient")
    print("##################################################\n")

    offset = 1

    for i in range(2*N):

        print(f"grad[{i}] = {sp.pycode(reduced[offset+i])}")

    offset += 2*N

    print("\n##################################################")
    print("# Hessian")
    print("##################################################\n")

    k = offset

    for i in range(2*N):

        for j in range(2*N):

            print(f"H[{i},{j}] = {sp.pycode(reduced[k])}")

            k += 1


if __name__ == "__main__":

    generate_python_code(5)
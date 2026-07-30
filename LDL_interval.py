from intvalpy import Interval


def interval_ldlt(A):
    """
    Perform an interval LDL^T decomposition of a symmetric matrix A.

    Parameters
    ----------
    A : list of list of Interval
        Symmetric matrix with interval entries.

    Returns
    -------
    L : list of list of Interval
        Lower triangular matrix with unit diagonal.
    D : list of Interval
        Diagonal matrix as a list of intervals.

    Notes
    -----
    If the matrix is not positive definite, the function returns None.
    """

    n = len(A)

    L = [[Interval(0,0) for _ in range(n)] for _ in range(n)]
    D = [None]*n

    for i in range(n):
        L[i][i] = Interval(1,1)

    for k in range(n):

        d = A[k][k]

        for j in range(k):
            d -= L[k][j]*L[k][j]*D[j]

        #
        # Rigorous proof fails here
        #

        if d.inf <= 0: #note infimum not infinity
            return None

        D[k]=d

        for i in range(k+1,n):

            t=A[i][k]

            for j in range(k):
                t -= L[i][j]*D[j]*L[k][j]

            L[i][k]=t/D[k]

    return L,D


def _interval_contains(outer, inner):
    return outer.inf <= inner.inf and outer.sup >= inner.sup


def _interval_overlaps(a, b):
    return not (a.sup < b.inf or b.sup < a.inf)


def validate_interval_ldlt(A, L, D, require_reconstruction_overlap=True):
    """
    Validate structural and interval-consistency properties of an LDL^T result.

    Parameters
    ----------
    A : list[list[Interval]]
        Original symmetric interval matrix.
    L : list[list[Interval]]
        Candidate lower triangular factor with unit diagonal.
    D : list[Interval]
        Candidate diagonal intervals.
    require_reconstruction_overlap : bool
        If True, verify each entry of A overlaps with the reconstructed
        interval product L*D*L^T.

    Returns
    -------
    ok : bool
        True if all checks pass.
    errors : list[str]
        Human-readable validation failures.
    """

    errors = []
    n = len(A)

    if len(L) != n or any(len(row) != n for row in L):
        errors.append("L must be an n x n matrix matching A")

    if len(D) != n:
        errors.append("D must have length n matching A")

    if errors:
        return False, errors

    one = Interval(1, 1)
    zero = Interval(0, 0)

    for i in range(n):
        if not _interval_contains(L[i][i], one):
            errors.append(f"L[{i},{i}] does not contain 1")

        if D[i].inf <= 0:
            errors.append(f"D[{i}] is not strictly positive")

        for j in range(i + 1, n):
            if not _interval_contains(L[i][j], zero):
                errors.append(f"L[{i},{j}] is not upper-triangular zero")

    if require_reconstruction_overlap:
        for i in range(n):
            for j in range(n):
                reconstructed = Interval(0, 0)
                for k in range(min(i, j) + 1):
                    reconstructed += L[i][k] * D[k] * L[j][k]

                if not _interval_overlaps(reconstructed, A[i][j]):
                    errors.append(
                        f"A[{i},{j}] does not overlap reconstructed L*D*L^T"
                    )

    return len(errors) == 0, errors


def validate_interval_ldlt_result(A, result, require_reconstruction_overlap=True):
    """Convenience wrapper for interval_ldlt(...) output."""

    if result is None:
        return False, ["No decomposition result to validate"]

    L, D = result
    return validate_interval_ldlt(
        A,
        L,
        D,
        require_reconstruction_overlap=require_reconstruction_overlap,
    )
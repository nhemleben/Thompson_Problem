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
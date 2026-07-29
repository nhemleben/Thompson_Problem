import numpy as np
import pytest
import sys
from pathlib import Path
from intvalpy import Interval

sys.path.append(str(Path(__file__).resolve().parents[1]))

from LDL_interval import interval_ldlt


def is_verified_pd(A):
    return interval_ldlt(A) is not None


###########################################################
# Exact SPD
###########################################################

def test_identity():

    A = [
        [Interval(1,1), Interval(0,0)],
        [Interval(0,0), Interval(1,1)]
    ]

    assert is_verified_pd(A)


###########################################################
# Simple SPD
###########################################################

def test_small_spd():

    A = [
        [Interval(4,4), Interval(1,1)],
        [Interval(1,1), Interval(3,3)]
    ]

    assert is_verified_pd(A)


###########################################################
# Interval SPD
###########################################################

def test_interval_spd():

    A = [
        [Interval(3.9,4.1), Interval(0.9,1.1)],
        [Interval(0.9,1.1), Interval(2.9,3.1)]
    ]

    assert is_verified_pd(A)


###########################################################
# Negative diagonal
###########################################################

def test_negative_diagonal():

    A = [
        [Interval(-1,-1), Interval(0,0)],
        [Interval(0,0), Interval(1,1)]
    ]

    assert not is_verified_pd(A)


###########################################################
# Indefinite matrix
###########################################################

def test_indefinite():

    A = [
        [Interval(1,1), Interval(2,2)],
        [Interval(2,2), Interval(1,1)]
    ]

    assert not is_verified_pd(A)


###########################################################
# Interval crossing zero
###########################################################

def test_crossing_zero():

    A = [
        [Interval(0.5,1.5), Interval(0.9,1.1)],
        [Interval(0.9,1.1), Interval(0.5,1.5)]
    ]

    #
    # The enclosure contains indefinite matrices,
    # therefore verification should fail.
    #

    assert not is_verified_pd(A)


###########################################################
# Random SPD matrices
###########################################################

@pytest.mark.parametrize("n", [3,4,5,6])
def test_random_spd(n):

    np.random.seed(1234+n)

    X = np.random.randn(n,n)

    A = X.T @ X + np.eye(n)

    AI = []

    for i in range(n):

        row=[]

        for j in range(n):

            row.append(Interval(A[i,j],A[i,j]))

        AI.append(row)

    assert is_verified_pd(AI)


###########################################################
# Random interval perturbations
###########################################################

@pytest.mark.parametrize("n",[3,4,5])
def test_random_interval_spd(n):

    np.random.seed(100+n)

    X=np.random.randn(n,n)

    A=X.T@X+2*np.eye(n)

    eps=1e-6

    AI=[]

    for i in range(n):

        row=[]

        for j in range(n):

            row.append(
                Interval(A[i,j]-eps,
                         A[i,j]+eps)
            )

        AI.append(row)

    assert is_verified_pd(AI)



###########################################################################

def monte_carlo_test():

    np.random.seed(1)

    for n in range(3,20):

        for _ in range(1000):

            X=np.random.randn(n,n)

            A=X.T@X+np.eye(n)

            AI=[[Interval(A[i,j],A[i,j]) for j in range(n)]
                 for i in range(n)]

            #
            # Exact matrix is SPD
            #

            assert np.min(np.linalg.eigvalsh(A))>0

            #
            # Interval code should certify it
            #

            assert is_verified_pd(AI)
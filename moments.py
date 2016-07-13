import numpy as np
import mpmath
from mpmath import mp
from mpmath import binomial
from mpmath import mpf

import mpmath.libmp

mp.dps = 2000

def moment_i0(a, j):
    '''
    Calculates the j-th moment of phi_0

    Wavelet Analysis: The Scalable Structure of Information (2002)
    Howard L. Resnikoff, Raymond O.Jr. Wells
    page 262
    '''
    N = a.size

    if j == 0:
        return 1

    s = mpmath.nsum(lambda k: binomial(j,k)*moment_i0(a, k)*
            mpmath.nsum(lambda i: a[int(i)]*mpmath.power(i, j-k),
                        [0, N-1]),
            [0, j-1])

    _2 = mp.mpmathify(2)
    return s/(_2*(mpmath.power(_2, j)-1))


def moment(a, i, j):
    '''
    Calculates the j-th moment of phi_i

    Wavelet Analysis: The Scalable Structure of Information (2002)
    Howard L. Resnikoff, Raymond O.Jr. Wells
    page 262
    '''
    N = a.size

    assert j > -1

    if j == 0:
        return 1 if i == 0 else 0

    if i == 0:
        return moment_i0(a, j)

    return mpmath.nsum(lambda k: binomial(j,k)*mpmath.power(i, (j-k))*moment_i0(a, k), [0, j])


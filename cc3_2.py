import itertools
import warnings

import numpy as np
from scipy.misc import factorial

from moments import moment
from scaling_coefficients import db3, daubechies_wavelets

"""
Here we calculate the 3-term connection coefficients

$$
Λ_{l,m}^{d,e} = ∫_{-∞}^∞ φ(x) φ^{(d)}_l(x) φ^{(e)}_m(x) dx
$$

with
$$
    -(N-1) < l < N-1, \\
    -(N-1) < m < N-1, \\
    -(N-1) < l-m < N-1.
$$
"""

def threeterm_connection_coefficients(a, d1, d2, d3):
    """
    Restriction to Fundamental Connection Coefficients
    --------------------------------------------------
    <a name="fundamental_cc"></a>
    """
    """
    The 3-term connection coefficients are defined as
    $$
    Λ_{i,j,k}^{d_1,d_2,d_3} = ∫^∞_{-∞} \frac{d^{d_1}}{dx^{d_1}} φ_i(x) \frac{d^{d_2}}{dx^{d_2}} φ_j(x) \frac{d^{d_3}}{dx^{d_3}} φ_k(x) dx
    $$
    which can be transformed by a change of variables with \(l=j-i\) and \(m=k-i\) to
    \begin{align}
    =& ∫^∞_{-∞} φ^{(d_1)}(x-i) φ^{(d_2)}(x-j) φ^{(d_2)}(x-k) dx \\
    =& ∫^∞_{-∞} φ^{(d_1)}(x) φ^{(d_2)}_l(x) φ^{(d_2)}_m(x) dx \\
    =& Λ_{l,m}^{d_1,d_2,d_3}.
    \end{align}
    """
    if d1 == 0:
        idx, indices, Λ = fundamental_threeterm_connection_coefficients(a, d1, d2, d3)
        idx2 = lambda i,j,k: idx(j-i, k-i)
        return idx2, indices, Λ
    else:
        """
        Using integration by parts we can focus on the case \(d_1=0\)
        """
        idx1, indices1, Λ1 = threeterm_connection_coefficients(a, d1-1, d2+1, d3)
        idx2, indices2, Λ2 = threeterm_connection_coefficients(a, d1-1, d2, d3+1)
        assert indices1 == indices2
        return idx1, indices1, -Λ1 - Λ2

def fundamental_threeterm_connection_coefficients(a, d1, d2, d3):
    """
    Calculation of Fundamental Connection Coefficients
    ==================================================
    """
    """
    Our wavelet has \(N=2g\) non-zero scaling coefficients where \(g\) is the genus.
    """
    N = a.size 
    aindices = range(N)
    d = d1 + d2 + d3

    """
    The fundamental connection coefficients \(Λ_{l,m}^{d_1,d_2,d_3}\) are just non-zero for
    $$
    -(N-1) < l < N-1, \\
    -(N-1) < m < N-1, \\
    -(N-1) < l-m < N-1.
    $$
    """
    Tindices = list(set((l,m) for l,m in itertools.product(range(-(N-2), (N-2)+1), repeat=2)
                      if abs(l-m) < (N-1)))
    idx = lambda l,m: Tindices.index((l,m))
    M = 3*N**2 - 9*N + 7
    assert len(Tindices)

    """
    The Daubechies wavelet of genus \(g=N/2\) has just as many vanishing moments and
    we must not calculate higher derivatives than \(d < g\)!
    """
    if np.amax([d1,d2,d3]) >= N/2:
        msg = 'Calculation of connection coefficients for {},{},{} > g = N/2 is invalid!'.format(d1,d2,d3)
        warnings.warn(msg)

    """
    Consequences of Compactness
    ---------------------------

    We exploit the fact that
    $$
    φ(x) = ∑_{i=0}^{N-1} a_i φ(2x-i)
    $$
    which means for the connection coefficients using the chain rule
    \begin{align}
    Λ_{l,m}^d =& ∫_{-∞}^∞ φ^{d_1}(x) φ^{(d_2)}_l(x) φ^{(d_3)}_m(x) dx \\
          =& ∫_{-∞}^∞
              \left(\frac{d}{dx}\right)^{d_1} ∑_{i=0}^{N-1} a_i φ(2x    -i) \\
            & × \left(\frac{d}{dx}\right)^{d_2} ∑_{j=0}^{N-1} a_j φ(2(x-l)-j) \\
            & × \left(\frac{d}{dx}\right)^{d_3} ∑_{k=0}^{N-1} a_k φ(2(x-m)-k) dx \\
          =& ∑_{i,j,k=0}^{N-1} a_i a_j a_k ∫_{-∞}^∞
                       2^{d_1} φ^{(d_1)}(2x-i) 2^{d_2} φ^{(d_2)}(2x-2l-j) 2^{d_3} φ^{(d_3)}(2x-2m-k) dx \\
          =& 2^d ∑_{i,j,k=0}^{N-1} a_i a_j a_k ∫_{-∞}^∞
                       φ^{(d_1)}(2x-i) φ^{(d_2)}(2x-2l-j) φ^{(d_3)}(2x-2m-k) dx \\
    \end{align}
    with \(d = d_1 + d_2 + d_3\).

    Using a change of variables \(2x ↦ x\) and remembering \(∫ f(2x) = \frac{1}{2} ∫ f(x) dx\)
    we find 
    \begin{align}
    =& \frac{1}{2} 2^d ∑_{i,j,k=0}^{N-1} a_i a_j a_k ∫_{-∞}^∞
                 φ^{(d_1)}(x-i) φ^{(d_2)}(x-2l-j) φ^{(d_3)}(x-2m-k) dx \\
    =& 2^{d-1} ∑_{i,j,k=0}^{N-1} a_i a_j a_k ∫_{-∞}^∞
                 φ^{(d_1)}(x) φ^{(d_2)}(x-2l-j+i) φ^{(d_3)}(x-2m-k+i) dx \\
    =& 2^{d-1} ∑_{i,j,k=0}^{N-1} a_i a_j a_k ∫_{-∞}^∞
                 φ^{(d_1)}(x) φ^{(d_2)}_{2l+j-i}(x) φ^{(d_3)}_{2m+k-i}(x) dx \\
    =& 2^{d-1} ∑_{i,j,k=0}^{N-1} a_i a_j a_k Λ_{2l+j-i,2m+k-i}^{d_1,d_2,d_3}. \\
    \end{align}
    
    This gives a system of \(XXX\) equations of the form
    $$
    (A - 2^{1-d} I) Λ^{d_1,d_2,d_3} = T Λ^{d_1,d_2,d_3} = 0
    $$
    where \(A_{l,m;2l+j-i,2m+k-i} = ∑_{i,j,k=0}^{N-1} a_i a_j a_k\).
    """
    T = np.zeros([len(Tindices), len(Tindices)])
    
    for l,m in Tindices:
        for i,j,k in itertools.product(range(N), repeat=3):
            if (2*l+j-i, 2*m+k-i) not in Tindices:
                continue # skip the Λ which are zero anyway
            T[idx(l,m), idx(2*l+j-i, 2*m+k-i)] += a[i]*a[j]*a[k]

    T -= 2**(1-d)*np.eye(len(Tindices))
    b = np.zeros([len(Tindices)])

    """
    Consequences of Moment Equations
    ------------------------------------
    If we differentiate the moment equation
    $$
    x^q = ∑_{i=-∞}^∞ M_i^q φ_i (x)
    $$
    \(d_1\) times with \(q < d_1\), we yield the equation
    $$
    0 = ∑_{i=-∞}^∞ M_i^q φ^{(d_1)}_i(x).
    $$
    Then multiplying by \(φ_j^{(d_2)} φ_k^{(d_3)}\) for some fixed \(j,k\),
    and integrating, we gain
    \begin{align}
    0 &= ∑_{i=-∞}^∞ M_i^q ∫_{-∞}^∞ φ^{(d_1)}_i(x) φ^{(d_2)}_j(x) φ^{d_3}_k(x) \\
      &= ∑_{i=-∞}^∞ M_i^q ∫_{-∞}^∞ φ^{(d_1)}(x-i) φ^{(d_2)}(x-j) φ^{d_3}(x-k).
    \end{align}
    Finally, we perform a change of variables \(x-i ↦ x\)
    \begin{align}
      &= ∑_{i=-∞}^∞ M_i^q ∫_{-∞}^∞ φ^{(d_1)}(x) φ^{(d_2)}(x-j+i) φ^{(d_3)}(x-k+i) \\
      &= ∑_{i=-∞}^∞ M_i^q Λ^{d_1,d_2,d_3}_{j-i,k-i} \\
      &= ∑_{i=-(N-2)}^{N-2} M_i^q Λ^{d_1,d_2,d_3}_{j-i,k-i}. \\
    \end{align}
    Similar equations hold for \(φ_j^{(d_2)}\) and \(φ_k^{(d_3)}\).
    """
    M = np.zeros([d1, len(Tindices)])
    j = 0 if (d2 % 2) == 0 else 1
    k = 0 if (d3 % 2) == 0 else 1
    for q in range(d1):
        for i in range(-(N-2), (N-2)+1):
            if (j-i, k-i) in Tindices:
                M[q, idx(j-i, k-i)] += moment(a, i, q)
    A = np.vstack([T,M])
    b = np.hstack([b, np.zeros([d1])])

    M = np.zeros([d2, len(Tindices)])
    i = 0 if (d1 % 2) == 0 else 1
    k = 0 if (d3 % 2) == 0 else 1
    for q in range(d2):
        for j in range(-(N-2), (N-2)+1):
            if (j-i, k-i) in Tindices:
                M[q, idx(j-i, k-i)] += moment(a, j, q)
    A = np.vstack([A,M])
    b = np.hstack([b, np.zeros([d2])])

    M = np.zeros([d3, len(Tindices)])
    i = 0 if (d1 % 2) == 0 else 1
    j = 0 if (d2 % 2) == 0 else 1
    for q in range(d3):
        for k in range(-(N-2), (N-2)+1):
            if (j-i, k-i) in Tindices:
                M[q, idx(j-i, k-i)] += moment(a, k, q)
    A = np.vstack([A,M])
    b = np.hstack([b, np.zeros([d3])])

    """
    Normalization of the Coefficients
    ---------------------------------
    Finally we differentiate the moment equation
    $$
    x^{d_1} = ∑_{i=-∞}^∞ M_i^{d_1} φ_i (x)
    $$
    \(d_1\) times, yielding
    $$
    d_1! = ∑_{i=-∞}^∞ M_i^{d_1} φ_i^{(d_1)} (x).
    $$
    Similar equations hold for \(φ_j\) and \(φ_k\).
    Multiplying these equations and integrating gains
    $$
    d_1!d_2!d_3! = ∑_{i,j,k=-∞}^∞ M_i^{d_1} M_j^{d_2} M_k^{d_3}
                     ∫_{-∞}^∞ φ_i^{(d_1)}(x) φ_j^{(d_2)}(x) φ_k^{(d_3)}(x) dx.
    $$
    Again with a change of variables \(x-i ↦ x\) this yields
    \begin{align}
    d_1!d_2!d_3! &= ∑_{i,j,k=-∞}^∞ M_i^{d_1} M_j^{d_2} M_k^{d_3}
                      ∫_{-∞}^∞ φ^{(d_1)}(x) φ_{j-i}^{(d_2)}(x) φ_{k-i}^{(d_3)}(x) dx \\
                 &= ∑_{i,j,k=-∞}^∞ M_i^{d_1} M_j^{d_2} M_k^{d_3}
                      Λ^{d_1,d_2,d_3}_{j-i,k-i} \\
                 &= ∑_{i,j,k=-(N-2)}^{N-2} M_i^{d_1} M_j^{d_2} M_k^{d_3}
                      Λ^{d_1,d_2,d_3}_{j-i,k-i}.
    \end{align}
    """
    M = np.zeros([1, len(Tindices)])
    i = 0
    for j,k in itertools.product(range(-(N-2), (N-2)+1), repeat=2):
        if (j-i, k-i) in Tindices:
            M[0, idx(j-i, k-i)] += moment(a, j, d2)*moment(a, k, d3)
    A = np.vstack([A,M])
    b = np.hstack([b, [factorial(d1)*factorial(d2)*factorial(d3)]])

    Λ, residuals, rank, singular_values  = np.linalg.lstsq(A, b)

    if (residuals > 1e-30).any():
        msg = 'Residuals {} of connection coefficients exceed 10**-30!'.format(residuals)
        warnings.warn(msg)

    return idx, Tindices, Λ


def test():
    from test_cc3 import cc3_100
    idx, Tindices, Λ = threeterm_connection_coefficients(db3, 1, 0, 0)

    N = len(db3)
    for l,m in itertools.product(range(-(N-2), (N-2)+1), repeat=2):
        a = Λ[idx(0,l,m)] if abs(l-m) < (N-1) else 0
        b = cc3_100[(l,m)]

        print(l,m, a, b, (a-b)/(b if b != 0 else 1))

if __name__  == '__main__':
    test()


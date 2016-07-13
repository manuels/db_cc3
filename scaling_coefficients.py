import mpmath
from mpmath import mp
import numpy as np

_1 = mp.mpmathify('1')
_2 = mp.mpmathify('2')
_3 = mp.mpmathify('3')
_4 = mp.mpmathify('4')
_5 = mp.mpmathify('5')
_10 = mp.mpmathify('10')
_16 = mp.mpmathify('16')

sqrt3 = mp.sqrt(_3)
sqrt10 = mp.sqrt(_10)

haar = np.array([_1, _1])
db1 = haar
D2 = db1

db2 = [(_1+sqrt3)/_4,
       (_3+sqrt3)/_4,
       (_3-sqrt3)/_4,
       (_1-sqrt3)/_4]
db2 = np.array(db2)
D4 = db2

'''
db3 scaling coefficients
Wavelet Analysis: The Scalable Structure of Information (2002)
Howard L. Resnikoff, Raymond O.Jr. Wells
Table 10.3 page 263
'''
db3 = [( _1 +    sqrt10 +    mp.sqrt(_5 + _2*sqrt10))/_16,
       ( _5 +    sqrt10 + _3*mp.sqrt(_5 + _2*sqrt10))/_16,
       (_10 - _2*sqrt10 + _2*mp.sqrt(_5 + _2*sqrt10))/_16,
       (_10 - _2*sqrt10 - _2*mp.sqrt(_5 + _2*sqrt10))/_16,
       ( _5 +    sqrt10 - _3*mp.sqrt(_5 + _2*sqrt10))/_16,
       ( _1 +    sqrt10 -    mp.sqrt(_5 + _2*sqrt10))/_16]
db3 = np.array(db3)
D6 = db3

for a in [D2, D4, D6]:
    norm = mpmath.norm(a)
    assert mpmath.almosteq(norm, mp.sqrt(_2))

def wavelet_coefficients(a):
    # TODO: is this just true for Daubechies wavelets?
    N = len(a)
    return np.array([(-1)**k*a[N-1-k] for k in range(N)])

# maps from genus (i.e. vanishing moments) to the scaling coefficients 
daubechies_wavelets = {
    1: D2,
    2: D4,
    3: D6,
}


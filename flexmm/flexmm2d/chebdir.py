import numpy as np
import numba

n = 12

x, _ = np.polynomial.chebyshev.chebgauss(n)
V = np.polynomial.chebyshev.chebvander(x, n-1)
a = np.exp(np.sin(x))
b = np.cos(x)*a
c = np.linalg.solve(V, a)

def chebeval1(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x

# get derivative by differentiating series and evaluating sum
cp = np.polynomial.chebyshev.chebder(c)
d = np.zeros(n)
for i in range(n):
    d[i] = chebeval1(x[i], cp)
print('Standard way error:  {:0.2e}'.format(np.abs(b-d).max()))

def chebeval_d1(x, c):
    x2 = 2*x
    c0 = c[-2] + x2*c[-1]
    c1 = c[-1]
    d0 = 2*c[-1]
    d1 = 0.0
    for i in range(3, len(c)):
        # recursion for d
        dm = 2*c0 + x2*d0 - d1
        d1 = d0
        d0 = dm
        # recursion for c
        cm = c[-i] + x2*c0 - c1
        c1 = c0
        c0 = cm
    return c0 + x*d0 - d1

# get derivative by using direct summation
d = np.zeros(n)
for i in range(n):
    d[i] = chebeval_d1(x[i], c)
print('New way error grad:  {:0.2e}'.format(np.abs(b-d).max()))

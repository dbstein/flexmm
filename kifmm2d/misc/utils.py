import numpy as np

def random(N, rmin=0, rmax=1):
    return np.random.rand(N)*(rmax-rmin) + rmin
def random2(N, rmin=0, rmax=1):
    return random(N,rmin,rmax), random(N,rmin,rmax)

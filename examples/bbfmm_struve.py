from flexmm2d.bbfmm import BB_FMM
from flexmm2d.misc.utils import random2
import numpy as np
import numba
import time
import os
from scipy.special import struve, yn
from function_generator.function_generator import FunctionGenerator

"""
Demonstration of the FMM for the Struve Kernel

If N <= 50000, will do a direct sum and compare to this
Otherwise, will try to call FMMLIB2D through pyfmmlib2d
To compare to
If this fails, no comparison for correctness!

On my macbook pro N=50,000 takes the direct method ~7s, the FMM <1s
(with N_equiv=64, N_cutoff=500)
And gives error <5e-14
"""

cpu_num = int(os.cpu_count()/2)
numba.config.NUMBA_NUM_THREADS = cpu_num
import mkl
mkl.set_num_threads(cpu_num)

# Greens Function
def GF(x):
    Y = yn(1, x)
    S3 = struve(-3, x)
    S2 = struve(-2, x)
    return (x*(Y-S3) - 4*S2)/x**2

# fast version of Greens Function
gf = FunctionGenerator(GF, a=1e-30, b=1000, tol=1e-12)
# extract compilable
_gf = gf.get_base_function()

# Kernel
@numba.njit(fastmath=True)
def Kernel(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d = np.sqrt(dx**2 + dy**2)
    return _gf(d)

N_source = 1000*10
N_target = 1000*10

# construct some data to run FMM on
px = np.random.rand(N_source)*100
py = np.random.rand(N_source)*100
rx = np.random.rand(N_target)*100
ry = np.random.rand(N_target)*100
bbox = [-10, 110, -10, 110]

# maximum number of points in each leaf of tree for FMM
N_cutoff = 30
# number of modes in Chebyshev expansions
p = 12

# get random density
tau = (np.random.rand(N_source))

print('\nWacky Kernel, FMM with', N_source, 'source pts and', N_target, 'target pts.')
FMM = BB_FMM(px, py, Kernel, N_cutoff, p, bbox=bbox)
FMM.build_expansions(tau)
_ = FMM.source_evaluation(px, py)
_ = FMM.target_evaluation(rx, ry)

st = time.time()
print('')
FMM = BB_FMM(px, py, Kernel, N_cutoff, p, bbox=bbox, helper=FMM.helper)
print('flexmm2d precompute took:             {:0.1f}'.format((time.time()-st)*1000))
st = time.time()
FMM.build_expansions(tau)
tt = (time.time()-st)
print('flexmm2d generation took:             {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
st = time.time()
self_fmm_eval = FMM.source_evaluation(px, py)
tt = (time.time()-st)
print('flexmm2d source eval took:            {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
st = time.time()
target_fmm_eval = FMM.target_evaluation(rx, ry)
tt = (time.time()-st)
print('flexmm2d target eval took:            {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')

# get reference solution
reference = False
KA = FMM.helper.functions['kernel_apply']
KAS = FMM.helper.functions['kernel_apply_self']
if N_source*N_target <= 10000**2:
    # by Direct Sum
    self_reference_eval = np.zeros(N_source, dtype=float)
    KAS(px, py, tau, out=self_reference_eval)
    st = time.time()
    KAS(px, py, tau, out=self_reference_eval)
    time_self_eval = (time.time() - st)*1000
    target_reference_eval = np.zeros(N_target, dtype=float)
    KA(px, py, rx, ry, tau, out=target_reference_eval)
    st = time.time()
    KA(px, py, rx, ry, tau, out=target_reference_eval)
    time_target_eval = (time.time() - st)*1000
    print('\nDirect self evaluation took:        {:0.1f}'.format(time_self_eval))
    print('Direct target evaluation took:      {:0.1f}'.format(time_target_eval))
    reference = True

if reference:
    sscale = np.abs(self_reference_eval)
    tscale = np.abs(target_reference_eval)
    sscale[sscale < 1] = 1
    tscale[tscale < 1] = 1
    self_err = np.abs(self_fmm_eval - self_reference_eval)/sscale
    target_err = np.abs(target_fmm_eval - target_reference_eval)/tscale
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))

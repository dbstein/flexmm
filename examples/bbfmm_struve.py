import flexmm2d
import flexmm2d.kernel_functions
import flexmm2d.fmm as fmm
import flexmm2d.bbfmm
from flexmm2d.misc.utils import random2
import numpy as np
import numba
import time
import os
from scipy.special import struve, yn
from function_generator import FunctionGenerator

"""
Demonstration of the FMM for the Laplace Kernel

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

# jit compile internal numba functions
functions = flexmm2d.kernel_functions.get_functions(Kernel)
KF = functions['kernel_form']
KA = functions['kernel_apply']
KAS = functions['kernel_apply_self']

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
p = 20
Nequiv = p**2

# get random density
tau = (np.random.rand(N_source))

print('\nWacky Kernel, FMM with', N_source, 'source pts and', N_target, 'target pts.')

# get reference solution
reference = False
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

# do my FMM (once first, to compile functions...)
functions = flexmm2d.bbfmm.get_functions(functions)
functions = fmm.get_functions(functions)
functions = flexmm2d.bbfmm.wrap_functions(functions)
FMM = fmm.FMM(px[:20*N_cutoff], py[:20*N_cutoff], functions, Nequiv, N_cutoff)
flexmm2d.bbfmm.precompute(FMM, p)
FMM.general_precomputations()
FMM.build_expansions(tau)
_ = FMM.evaluate_to_points(px[:20*N_cutoff], py[:20*N_cutoff], True)

st = time.time()
print('')
FMM = fmm.FMM(px, py, functions, Nequiv, N_cutoff, bbox=bbox)
flexmm2d.bbfmm.precompute(FMM, p)
FMM.general_precomputations()
print('pyfmmlib2d precompute took:           {:0.1f}'.format((time.time()-st)*1000))
st = time.time()
FMM.build_expansions(tau)
tt = (time.time()-st)
print('pyfmmlib2d generation took:           {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
st = time.time()
self_fmm_eval = FMM.evaluate_to_points(px, py, True)
tt = (time.time()-st)
print('pyfmmlib2d source eval took:          {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')

st = time.time()
target_fmm_eval = FMM.evaluate_to_points(rx, ry)
tt = (time.time()-st)
print('pyfmmlib2d target eval took:          {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

if reference:
    sscale = np.abs(self_reference_eval)
    tscale = np.abs(target_reference_eval)
    sscale[sscale < 1] = 1
    tscale[tscale < 1] = 1
    self_err = np.abs(self_fmm_eval - self_reference_eval)/sscale
    target_err = np.abs(target_fmm_eval - target_reference_eval)/tscale
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))

from flexmm2d.bbfmm import BB_FMM
from flexmm2d.misc.utils import random2
import numpy as np
import numba
import time
import os

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

# Laplace Kernel
@numba.njit(fastmath=True)
def Laplace_Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx**2 + dy**2
    return -np.log(d2)/(4*np.pi)

N_source = 1000*100
N_target = 1000*100
test = 'circle' # clustered or circle or uniform

# construct some data to run FMM on
if test == 'uniform':
    px = np.random.rand(N_source)
    py = np.random.rand(N_source)
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
    bbox = None
elif test == 'clustered':
    N_clusters = 10
    N_per_cluster = int((N_source / N_clusters))
    N_random = N_source - N_clusters*N_per_cluster
    center_clusters_x, center_clusters_y = random2(N_clusters, -99, 99)
    px, py = random2(N_source, -1, 1)
    px[:N_random] *= 100
    py[:N_random] *= 100
    px[N_random:] += np.repeat(center_clusters_x, N_per_cluster)
    py[N_random:] += np.repeat(center_clusters_y, N_per_cluster)
    px /= 100
    py /= 100
    rx = np.random.rand(N_target)
    ry = np.random.rand(N_target)
    bbox = [0,1,0,1]
elif test == 'circle':
    rand_theta = np.random.rand(int(N_source))*2*np.pi
    px = np.cos(rand_theta)
    py = np.sin(rand_theta)
    rx = (np.random.rand(N_target)-0.5)*10
    ry = (np.random.rand(N_target)-0.5)*10
    bbox = [-5,5,-5,5]
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 50
# number of modes in Chebyshev expansions
p = 12

# get random density
tau = (np.random.rand(N_source))

print('\nLaplace FMM with', N_source, 'source pts and', N_target, 'target pts.')

# get reference solution
reference = True
if reference:
    if N_source*N_target <= 10000**2:
        self_reference_eval = np.zeros(N_source, dtype=complex)
        KAS(px, py, tau, out=self_reference_eval)
        # by Direct Sum
        st = time.time()
        KAS(px, py, tau, out=self_reference_eval)
        time_self_eval = (time.time() - st)*1000
        target_reference_eval = np.zeros(N_target, dtype=complex)
        KA(px, py, rx, ry, tau, out=target_reference_eval)
        st = time.time()
        KA(px, py, rx, ry, tau, out=target_reference_eval)
        time_target_eval = (time.time() - st)*1000
        print('\nDirect self evaluation took:        {:0.1f}'.format(time_self_eval))
        print('Direct target evaluation took:      {:0.1f}'.format(time_target_eval))
    else:
        # by FMMLIB2D, if available
        try:
            import pyfmmlib2d
            source = np.row_stack([px, py])
            target = np.row_stack([rx, ry])
            dumb_targ = np.row_stack([np.array([0.6, 0.6]), np.array([0.5, 0.5])])
            st = time.time()
            out = pyfmmlib2d.RFMM(source, dumb_targ, charge=tau, compute_target_potential=True)
            tform = time.time() - st
            print('FMMLIB generation took:               {:0.1f}'.format(tform*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tform/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True)
            self_reference_eval = -0.5*out['source']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB self only eval took:           {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, target, charge=tau, compute_target_potential=True)
            target_reference_eval = -0.5*out['target']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB target only eval took:         {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')
        except:
            print('')
            reference = False

FMM = BB_FMM(px, py, Laplace_Eval, N_cutoff, p, bbox=bbox)
FMM.build_expansions(tau)
_ = FMM.source_evaluation(px[:20*N_cutoff], py[:20*N_cutoff])
_ = FMM.target_evaluation(rx[:20*N_cutoff], ry[:20*N_cutoff])

st = time.time()
print('')
FMM = BB_FMM(px, py, Laplace_Eval, N_cutoff, p, bbox=bbox, helper=FMM.helper)
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
print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

if reference:
    sscale = np.abs(self_reference_eval).max()
    tscale = np.abs(target_reference_eval).max()
    self_err = np.abs(self_fmm_eval - self_reference_eval)/sscale
    target_err = np.abs(target_fmm_eval - target_reference_eval)/tscale
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))

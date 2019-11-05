import flexmm2d
import flexmm2d.kernel_functions
import flexmm2d.fmm as fmm
import flexmm2d.kifmm
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

# Laplace Kernel
@numba.njit(fastmath=True)
def Laplace_Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    return -np.log(d2)/(4*np.pi)

# Laplace Gradient Kernel
@numba.njit(fastmath=True)
def Laplace_Gradient_Eval(sx, sy, tx, ty):
    dx = tx-sx
    dy = ty-sy
    d2 = dx*dx + dy*dy
    id2 = 1.0/d2
    scale = -1.0/(2*np.pi)
    u = -np.log(d2)/(4*np.pi)
    ux = scale*dx*id2
    uy = scale*dy*id2
    return u, ux, uy

# Laplace Dipole Kernel
@numba.njit(fastmath=True)
def Laplace_Dipole_Eval(sx, sy, tx, ty, nx, ny):
    dx = tx-sx
    dy = ty-sy
    n_dot_d = nx*dx + ny*dy
    d2 = dx*dx + dy*dy
    return n_dot_d/d2/(2*np.pi)

# jit compile internal numba functions
functions = flexmm2d.kernel_functions.get_functions(Laplace_Eval)
functions = flexmm2d.kernel_functions.add_gradient_functions(Laplace_Gradient_Eval, functions)
KF = functions['kernel_form']
KA = functions['kernel_apply']
KAS = functions['kernel_apply_self']

N_source = 1000*10
N_target = 1000*1000
test = 'circle' # clustered or circle or uniform
reference_precision = 2

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
    bbox = [-5,5,5,5]
else:
    raise Exception('Test is not defined')

# maximum number of points in each leaf of tree for FMM
N_cutoff = 100
# number of modes in source/check surfaces
Nequiv = 30

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
            out = pyfmmlib2d.RFMM(source, dumb_targ, charge=tau, compute_target_potential=True, precision=reference_precision)
            tform = time.time() - st
            print('FMMLIB generation took:               {:0.1f}'.format(tform*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tform/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, charge=tau, compute_source_potential=True, precision=reference_precision)
            self_reference_eval = -0.5*out['source']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB self only eval took:           {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, target, charge=tau, compute_target_potential=True, precision=reference_precision)
            target_reference_eval = -0.5*out['target']['u']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB target only eval took:         {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')
            st = time.time()
            out = pyfmmlib2d.RFMM(source, target, charge=tau, compute_target_potential=True, compute_target_gradient=True, precision=reference_precision)
            target_gradient_eval_x = -0.5*out['target']['u_x']/np.pi
            target_gradient_eval_y = -0.5*out['target']['u_y']/np.pi
            tt = time.time() - st - tform
            print('FMMLIB target pot/grad eval took:     {:0.1f}'.format(tt*1000))
            print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')
        except:
            print('')
            reference = False

# do my FMM (once first, to compile functions...)
functions = flexmm2d.kifmm.get_functions(functions)
functions = fmm.get_functions(functions)

FMM = fmm.FMM(px[:20*N_cutoff], py[:20*N_cutoff], functions, Nequiv, N_cutoff)
precomputations = flexmm2d.kifmm.KI_Precomputations(FMM)
FMM.load_precomputations(precomputations)
FMM.build_expansions(tau)
_ = FMM.evaluate_to_points(px[:20*N_cutoff], py[:20*N_cutoff], True)
_ = FMM.evaluate_gradient_to_points(px[:20*N_cutoff], py[:20*N_cutoff], True)

print('')
st = time.time()
FMM = fmm.FMM(px, py, functions, Nequiv, N_cutoff, bbox=bbox)
print('flexmm2d setup took:                  {:0.1f}'.format((time.time()-st)*1000))
precomputations = flexmm2d.kifmm.KI_Precomputations(FMM)
FMM.load_precomputations(precomputations)
print('flexmm2d precompute took:             {:0.1f}'.format((time.time()-st)*1000))
st = time.time()
FMM.build_expansions(tau)
tt = (time.time()-st)
print('flexmm2d generation took:             {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')
st = time.time()
self_fmm_eval = FMM.evaluate_to_points(px, py, True)
tt = (time.time()-st)
print('flexmm2d source eval took:            {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_source/tt/cpu_num/1000), '\033[0m ')

st = time.time()
target_fmm_eval = FMM.evaluate_to_points(rx, ry)
tt = (time.time()-st)
print('flexmm2d target eval took:            {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')
st = time.time()
target_fmm_grad = FMM.evaluate_gradient_to_points(rx, ry)
tt = (time.time()-st)
print('flexmm2d target pot/grad eval took:   {:0.1f}'.format(tt*1000))
print('...Points/Second/Core (thousands)    \033[1m', int(N_target/tt/cpu_num/1000), '\033[0m ')

if reference:
    sscale = np.abs(self_reference_eval).max()
    tscale = np.abs(target_reference_eval).max()
    tx_scale = np.abs(target_gradient_eval_x).max()
    ty_scale = np.abs(target_gradient_eval_y).max()
    self_err = np.abs(self_fmm_eval - self_reference_eval)/sscale
    target_err = np.abs(target_fmm_eval - target_reference_eval)/tscale
    target_err_x = np.abs(target_fmm_grad[1] - target_gradient_eval_x)/tx_scale
    target_err_y = np.abs(target_fmm_grad[2] - target_gradient_eval_y)/ty_scale
    target_err_grad = max(target_err_x.max(), target_err_y.max())
    print('\nMaximum difference, self:             {:0.2e}'.format(self_err.max()))
    print('Maximum difference, target:           {:0.2e}'.format(target_err.max()))
    print('Maximum difference, targ-gradinet:    {:0.2e}'.format(target_err.max()))


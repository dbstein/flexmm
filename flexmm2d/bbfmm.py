import numpy as np
import numba
import scipy as sp
import scipy.linalg
from .fmm import FMM
from .fmm import get_functions as fmm_get_functions
from .precomputations import Precomputation, Precomputations

"""
Define necessary functions and precomputations for BB-Style FMM
"""

def BBFMM_PREP(functions):
    functions = get_functions(functions)
    functions = fmm_get_functions(functions)
    functions = wrap_functions(functions)
    return functions

def BBFMM(x, y, functions, tau, p=10, Ncutoff=50, iscomplex=False, bbox=None, verbose=False):
    fmm = FMM(x, y, functions, p**2, Ncutoff, iscomplex, bbox, verbose)
    precompute(fmm, p)
    fmm.general_precomputations()
    fmm.build_expansions(tau)
    return fmm

############################################################################
# These are helper functions for this particular FMM implementation

@numba.njit(parallel=True, fastmath=True)
def anterpolate2(p, x, y, tau):
    return chebV2(x, y, p).T.dot(tau)
@numba.njit(parallel=True, fastmath=True)
def chebV2(xs, ys, p):
    xv = all_chebt(p, xs)
    yv = all_chebt(p, ys)
    out = np.empty((xs.size, p, p))
    for i in numba.prange(xs.size):
        for j in range(p):
            for k in range(p):
                out[i,j,k] = xv[i,j]*yv[i,k]
    return out.reshape(xs.size, p**2)
@numba.njit(fastmath=True)
def chebeval2d1(x, y, c):
    p = c.shape[0]
    w1 = np.empty(p, c.dtype)
    for i in range(p):
        w1[i] = chebeval1(x, c[:,i])
    return chebeval1(y, w1)
@numba.njit(fastmath=True)
def chebeval2d_dx(x, y, c):
    p = c.shape[0]
    w1 = np.empty(p, c.dtype)
    for i in range(p):
        w1[i] = chebeval_d1(x, c[:,i])
    return chebeval1(y, w1)
@numba.njit(fastmath=True)
def chebeval2d_dy(x, y, c):
    p = c.shape[0]
    w1 = np.empty(p, c.dtype)
    for i in range(p):
        w1[i] = chebeval1(x, c[:,i])
    return chebeval_d1(y, w1)
@numba.njit(parallel=True, fastmath=True)
def all_chebt(p, x):
    out = np.empty((x.size, p))
    for i in numba.prange(x.size):
        out[i] = all_chebt1(p, x[i])
    return out
@numba.njit(fastmath=True)
def all_chebt1(p, x):
    out = np.empty(p)
    out[0] = 1.0
    if p > 1:
        out[1] = x
    if p > 2:
        x2 = 2*x
        c0 = 0
        c1 = 1
        for i in range(2, p):
            tmp = c0
            c0 = 0 - c1
            c1 = tmp + c1*x2
            out[i] = c0 + c1*x
    return out
@numba.njit(fastmath=True)
def chebeval1(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x
@numba.njit(fastmath=True)
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

def get_functions(functions):

    kernel_apply_single       = functions['kernel_apply_single']

    ############################################################################
    # These functions DEPEND on the particular FMM implementation

    @numba.njit(fastmath=True)
    def source_to_partial_multipole(sx, sy, tau, expansion, width_adj, p):
        expansion[:] = anterpolate2(p, sx*width_adj, sy*width_adj, tau)

    def partial_multipole_to_multipole(pM, dummy):
        return pM

    def partial_local_to_local(pL, dummy):
        return pL

    @numba.njit(fastmath=True)
    def local_expansion_to_target(expansion, tx, ty, width_adj, p):
        return chebeval2d1(tx*width_adj, ty*width_adj, expansion.reshape(p,p))

    new_functions = {
        'partial_multipole_to_multipole' : partial_multipole_to_multipole,
        'partial_local_to_local'         : partial_local_to_local,
        'source_to_partial_multipole'    : source_to_partial_multipole,
        'local_expansion_to_target'      : local_expansion_to_target,
        'chebV2'                         : chebV2,
    }
    functions.update(new_functions)

    return functions

class BB_Precomputations(Precomputations):
    def __init__(self, fmm, precomputations=None):
        super().__init__(precomputations)

        if hasattr(precomputations, 'info'):
            self.info = precomputations.info
        else:
            p = int(np.round(np.sqrt(fmm.Nequiv)))
            # some basic chebyshev stuff
            nodexv = np.polynomial.chebyshev.chebgauss(p)[0][::-1]
            nodeyv = np.polynomial.chebyshev.chebgauss(p)[0][::-1]
            nodex, nodey = np.meshgrid(nodexv, nodeyv, indexing='ij')
            nodex, nodey = nodex.ravel(), nodey.ravel()
            V = np.polynomial.chebyshev.chebvander2d(nodex, nodey, [p-1, p-1])
            VI = np.linalg.inv(V)

            # get the M2MC for each level (always the same!)
            snodex, snodey = nodex/2, nodey/2
            snodex00 = snodex - 0.5
            snodey00 = snodey - 0.5
            snodex01 = snodex - 0.5
            snodey01 = snodey + 0.5
            snodex10 = snodex + 0.5
            snodey10 = snodey - 0.5
            snodex11 = snodex + 0.5
            snodey11 = snodey + 0.5
            collected_snodex = np.concatenate([snodex00, snodex01, snodex10, snodex11])
            collected_snodey = np.concatenate([snodey00, snodey01, snodey10, snodey11])
            W1 = np.zeros([4*p**2, 4*p**2])
            for i in range(4):
                W1[i*p**2:(i+1)*p**2, i*p**2:(i+1)*p**2] = VI.T
            W2 = chebV2(collected_snodex, collected_snodey, p).T
            M2MC = W2.dot(W1)
            # get L2L operators
            L2LC = M2MC.T

            self.info = {
                'nodex' : nodex,
                'nodey' : nodey,
                'VI'    : VI,
                'M2MC'  : M2MC,
                'L2LC'  : L2LC,
            }

        self.prepare(fmm)

    def prepare(self, fmm):
        tree = fmm.tree
        self.width_adjs = []
        self.ps = []
        for Level in tree.Levels:
            width = Level.width
            if not self.has_precomputation(width):
                precomp = BB_Precomputation(width, fmm, self.info)
                self.add_precomputation(precomp)
            self.width_adjs.append(self[width].width_adj)
            self.ps.append(self[width].p)
    def get_local_expansion_extras(self):
        return self.width_adjs, self.ps

class BB_Precomputation(Precomputation):
    def __init__(self, width, fmm, info):
        super().__init__(width)
        self.precompute(fmm, info)
        self.compress()

    def precompute(self, fmm, info):
        tree = fmm.tree
        width = self.width
        p = int(np.round(np.sqrt(fmm.Nequiv)))
        KF = fmm.functions['kernel_form']
        dtype = fmm.dtype

        nodex = info['nodex']
        nodey = info['nodey']
        VI    = info['VI']
        self.M2MC  = info['M2MC']
        self.L2LC  = info['L2LC']

        # get all required M2L translations
        M2L = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2L[indx, indy] = None
                else:
                    snodex = nodex*0.5*width
                    snodey = nodey*0.5*width
                    translated_snodex = snodex + (indx-3)*width
                    translated_snodey = snodey + (indy-3)*width
                    W1 = KF(translated_snodex, translated_snodey, snodex, snodey, mdtype=dtype)
                    M2L[indx,indy] = VI.dot(W1).dot(VI.T)
        self.M2L = M2L
        # used for rescaling things
        self.width_adj = 2.0/width
        self.p = p
    def get_upwards_extras(self):
        return self.width_adj, self.p
    def get_partial_multipole_to_multipole_extra(self):
        return None
    def get_partial_local_to_local_extra(self):
        return None

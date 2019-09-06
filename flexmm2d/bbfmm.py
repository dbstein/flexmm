import numpy as np
import numba
import scipy as sp
import scipy.linalg
from .fmm import FMM
from .fmm import get_functions as fmm_get_functions

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

def Kernel_Form(KF, sx, sy, tx=None, ty=None, out=None, mdtype=float):
    if tx is None or ty is None:
        tx = sx
        ty = sy
        isself = True
    else:
        if sx is tx and sy is ty:
            isself = True
        else:
            isself = False
    ns = sx.shape[0]
    nt = tx.shape[0]
    if out is None:
        out = np.empty((nt, ns), dtype=mdtype)
    KF(sx, sy, tx, ty, out)
    if isself:
        np.fill_diagonal(out, 0.0)
    return out

def wrap_functions(functions):

    upwards_pass               = functions['upwards_pass']
    local_expansion_evaluation = functions['local_expansion_evaluation']

    def wrapped_upwards_pass(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, precomputations, ind):
        upwards_pass(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, \
                precomputations['width_adj'][ind], precomputations['p'])

    def wrapped_local_expansion_evaluation(x, y, inds, locs, xmids, ymids, Local_Expansions, pot, precomputations):
        local_expansion_evaluation(x, y, inds, locs, xmids, ymids, Local_Expansions, pot, \
                precomputations['width_adj'], precomputations['ps'])

    new_functions = {
        'wrapped_upwards_pass'               : wrapped_upwards_pass,
        'wrapped_local_expansion_evaluation' : wrapped_local_expansion_evaluation,
    }

    functions.update(new_functions)
    return functions

def get_functions(functions):

    kernel_add                = functions['kernel_add']
    kernel_apply_single       = functions['kernel_apply_single']

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

    ############################################################################
    # These functions DEPEND on the particular FMM implementation

    @numba.njit(fastmath=True)
    def source_to_partial_multipole(sx, sy, tau, expansion, width_adj, p):
        expansion[:] = anterpolate2(p, sx*width_adj, sy*width_adj, tau)

    def partial_multipole_to_multipole(pM, precomputations, ind):
        return pM

    def partial_local_to_local(pL, precomputations, ind):
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

def precompute(fmm, p):
    """
    Precomputations for KI-Style FMM
    kwargs:
        required:
            Nequiv: int, number of points used in equivalent surfaces
    """
    tree = fmm.tree
    Ncutoff = fmm.Ncutoff
    KF = fmm.functions['kernel_form']
    chebV2 = fmm.functions['chebV2']

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
    M2M = W2.dot(W1)
    # get level widths (these are important!)
    widths = []
    for Level in tree.Levels:
        widths.append(Level.width)
    # get Collected Equivalent Coordinates for each level
    M2MC = []
    for ind in range(tree.levels):
        M2MC.append(M2M)
    # get L2LC operators
    L2LC = [A.T for A in M2MC]
    # get all required M2L translations
    M2LS = []
    M2LS.append(None)
    for ind in range(1, tree.levels):
        M2Lhere = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2Lhere[indx, indy] = None
                else:
                    snodex = nodex*0.5*widths[ind]
                    snodey = nodey*0.5*widths[ind]
                    translated_snodex = snodex + (indx-3)*widths[ind]
                    translated_snodey = snodey + (indy-3)*widths[ind]
                    W1 = Kernel_Form(KF, translated_snodex, translated_snodey, snodex, snodey, mdtype=fmm.dtype)
                    M2Lhere[indx,indy] = VI.dot(W1).dot(VI.T)
        M2LS.append(M2Lhere)

    precomputations = {
        'M2MC'      : M2MC,
        'L2LC'      : L2LC,
        'M2LS'      : M2LS,
        'p'         : p,
        'widths'    : np.array(widths).astype(float),
        'width_adj' : 2.0/np.array(widths).astype(float),
        'null'      : [None,]*tree.levels,
        'ps'        : np.repeat(p, tree.levels),
    }

    fmm.precomputations = precomputations

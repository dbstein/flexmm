import numpy as np
import numba
import scipy as sp
import scipy.linalg
from .fmm import FMM
from .helpers import Helper, Precomputation

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

class BB_Precomputation(Precomputation):
    def __init__(self, width, fmm, info):
        super().__init__(width)
        self.precompute(fmm, info)
        self.compress()
    def precompute(self, fmm, info):
        self.M2MC = info['M2MC']
        self.L2LC = info['L2LC']
        VI = info['VI']
        nodex = info['nodex']
        nodey = info['nodey']

        tree = fmm.tree
        width = self.width
        KF = fmm.helper.functions['kernel_form']
        dtype = fmm.dtype

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
        self.p = fmm.p

class BB_Helper(Helper):
    def __init__(self, helper=None):
        super().__init__(helper)
    def prepare(self, fmm):
        if not hasattr(self, 'info'):
            p = fmm.p
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

        tree = fmm.tree
        self.width_adjs = []
        for Level in tree.Levels:
            width = Level.width
            if not self.has_precomputation(width):
                precomp = BB_Precomputation(width, fmm, self.info)
                self.add_precomputation(precomp)
            precomp = self.get_precomputation(width)
            self.width_adjs.append(precomp.width_adj)

    def build_s2pM_function(self):
        if 'source2pMs' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def source2pMs(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, p, width_adj):
                for i in numba.prange(bind.size):
                    if cu[i] and li[i] >= 0:
                        bi = bind[i]
                        ti = tind[i]
                        sx = x[bi:ti]-xmid[i]
                        sy = y[bi:ti]-ymid[i]
                        pM[li[i],:] = anterpolate2(p, sx*width_adj, sy*width_adj, tau[bi:ti])
            self.functions['source2pMs'] = source2pMs
    def build_l2t_function(self):
        kernel_apply_single = self.functions['kernel_apply_single']
        
        if 'L2targets' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def local_expansion_evaluation(tx, ty, inds, locs, xmids, ymids, LEs, pot, p, width_adjs):
                for i in numba.prange(tx.size):
                    x = tx[i]
                    y = ty[i]
                    ind = inds[i]
                    loc = locs[i]
                    x = x - xmids[ind][loc]
                    y = y - ymids[ind][loc]
                    width_adj = width_adjs[ind]
                    expansion = LEs[ind][loc]
                    pot[i] = chebeval2d1(x*width_adj, y*width_adj, expansion.reshape(p,p))
            self.functions['L2targets'] = local_expansion_evaluation

class BB_FMM(FMM):
    def __init__(self, x, y, kernel_eval, Ncutoff=50, p=8, dtype=float, bbox=None, helper=BB_Helper(), verbose=False):
        super().__init__(x, y, kernel_eval, Ncutoff, dtype, bbox, helper, verbose)
        self.p = p
        self.Nequiv = self.p**2
        self.helper.build_s2pM_function()
        self.helper.build_l2t_function()
        self.helper.prepare(self)
    def source_to_partial_multipole(self, ind, partial_multipole):
        tree = self.tree
        Level = tree.Levels[ind]
        tau_ordered = self.tau_ordered
        prec = self.helper.get_precomputation(Level.width)
        self.helper.functions['source2pMs'](tree.x, tree.y, Level.this_density_ind,
            Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid,
            tau_ordered, partial_multipole, self.p, prec.width_adj)
    def local_to_targets(self, x, y, pot, inds, locs):
        tree = self.tree
        self.helper.functions['L2targets'](x, y, inds, locs, tree.xmids, tree.ymids,
            self.Local_Expansions, pot, self.p, self.helper.width_adjs)

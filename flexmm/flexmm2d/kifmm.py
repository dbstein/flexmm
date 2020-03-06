import numpy as np
import numba
import scipy as sp
import scipy.linalg
from .fmm import FMM
from .helpers import Helper, Precomputation

try:
    from numba.typed import List
except:
    List = list

class KI_Precomputation(Precomputation):
    def __init__(self, width, fmm):
        super().__init__(width)
        self.precompute(fmm)
        self.compress()
    def precompute(self, fmm):
        width = self.width
        Nequiv = fmm.Nequiv
        KF = fmm.helper.functions['kernel_form']
        dtype = fmm.dtype
        theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
        sx, sy, lx, ly, sr, lr = get_level_information(width, theta, Nequiv)
        small_to_large = KF(sx, sy, lx, ly, mdtype=dtype)
        self.S2L_LU = sp.linalg.lu_factor(small_to_large, overwrite_a=True, check_finite=False)
        large_to_small = KF(sx, sy, lx, ly, mdtype=dtype)
        self.L2S_LU = sp.linalg.lu_factor(large_to_small, overwrite_a=True, check_finite=False)
        # get Collected Equivalent Coordinates for the smaller level
        ssx, ssy, _, _, _, _ = get_level_information(0.5*width, theta, Nequiv)
        cexs = np.concatenate([
                ssx - 0.25*width,
                ssx - 0.25*width,
                ssx + 0.25*width,
                ssx + 0.25*width,
            ])
        ceys = np.concatenate([
                ssy - 0.25*width,
                ssy + 0.25*width,
                ssy - 0.25*width,
                ssy + 0.25*width,
            ])
        # get M2MC operator
        self.M2MC = KF(cexs, ceys, lx, ly, mdtype=dtype)
        # get L2LC operator
        self.L2LC = self.M2MC.T
        # get all required M2L translations
        M2L = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2L[indx, indy] = None
                else:
                    sx_here = sx + (indx - 3)*width
                    sy_here = sy + (indy - 3)*width
                    M2L[indx,indy] = KF(sx_here, sy_here, sx, sy, mdtype=dtype)
        self.M2L = M2L
        # stow away some other things
        self.small_x = sx
        self.small_y = sy
        self.large_x = lx
        self.large_y = ly

class KI_Helper(Helper):
    def __init__(self, helper=None):
        super().__init__(helper)
    def prepare(self, fmm):
        tree = fmm.tree
        self.small_xs = List()
        self.small_ys = List()
        self.large_xs = List()
        self.large_ys = List()
        for Level in tree.Levels:
            width = Level.width
            if not self.has_precomputation(width):
                precomp = KI_Precomputation(width, fmm)
                self.add_precomputation(precomp)
            precomp = self.get_precomputation(width)
            self.small_xs.append(precomp.small_x)
            self.small_ys.append(precomp.small_y)
            self.large_xs.append(precomp.large_x)
            self.large_ys.append(precomp.large_y)
    def build_s2pM_function(self):
        Kernel_Eval = self.functions['kernel_eval']
        if 'kernel_apply' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def kernel_apply(sx, sy, tx, ty, tau, out):
                for j in numba.prange(tx.size):
                    outj = 0.0
                    for i in range(sx.size):
                        outj += Kernel_Eval(sx[i], sy[i], tx[j], ty[j])*tau[i]
                    out[j] = outj
            self.functions['kernel_apply'] = kernel_apply
        kernel_apply = self.functions['kernel_apply']

        if 'source2pMs' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def source2pMs(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, large_x, large_y):
                for i in numba.prange(bind.size):
                    if cu[i] and li[i] >= 0:
                        bi = bind[i]
                        ti = tind[i]
                        kernel_apply(x[bi:ti]-xmid[i],
                            y[bi:ti]-ymid[i], large_x, large_y, tau[bi:ti], pM[li[i]])
            self.functions['source2pMs'] = source2pMs
    def build_l2t_function(self):
        kernel_apply_single = self.functions['kernel_apply_single']
        
        if 'L2targets' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def local_expansion_evaluation(tx, ty, inds, locs, xmids, ymids, LEs, pot, large_xs, large_ys):
                for i in numba.prange(tx.size):
                    x = tx[i]
                    y = ty[i]
                    ind = inds[i]
                    loc = locs[i]
                    x = x - xmids[ind][loc]
                    y = y - ymids[ind][loc]
                    pot[i] = kernel_apply_single(large_xs[ind], large_ys[ind], x, y, LEs[ind][loc])
            self.functions['L2targets'] = local_expansion_evaluation

class KI_FMM(FMM):
    def __init__(self, x, y, kernel_eval, Ncutoff=50, Nequiv=50, dtype=float, bbox=None, helper=KI_Helper(), verbose=False):
        super().__init__(x, y, kernel_eval, Ncutoff, dtype, bbox, helper, verbose)
        self.Nequiv = Nequiv
        self.helper.build_s2pM_function()
        self.helper.build_l2t_function()
        self.helper.prepare(self)
    def partial_multipole_to_multipole(self, prec, pM):
        return sp.linalg.lu_solve(prec.S2L_LU, pM.T, overwrite_b=True, check_finite=False).T
    def partial_local_to_local(self, prec, pL):
        return sp.linalg.lu_solve(prec.L2S_LU, pL.T, overwrite_b=True, check_finite=False).T
    def source_to_partial_multipole(self, ind, partial_multipole):
        tree = self.tree
        Level = tree.Levels[ind]
        tau_ordered = self.tau_ordered
        prec = self.helper.get_precomputation(Level.width)
        self.helper.functions['source2pMs'](tree.x, tree.y, Level.this_density_ind,
            Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid,
            tau_ordered, partial_multipole, prec.large_x, prec.large_y)
    def local_to_targets(self, x, y, pot, inds, locs):
        tree = self.tree
        self.helper.functions['L2targets'](x, y, inds, locs, tree.xmids, tree.ymids,
            self.Local_Expansions, pot, self.helper.large_xs, self.helper.large_ys)

"""
Define necessary functions and precomputations for KI-Style FMM
"""

def get_functions(functions):

    @numba.njit(fastmath=True)
    def local_expansion_to_target(expansion, tx, ty, sx, sy):
        return kernel_apply_single(sx, sy, tx, ty, expansion)

    return functions

def get_level_information(node_width, theta, N):
    # get information for this level
    dd = 0.0
    r1 = 0.5*node_width*(np.sqrt(2)+dd)
    r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
    small_surface_x_base = r1*np.cos(theta)
    small_surface_y_base = r1*np.sin(theta)
    large_surface_x_base = r2*np.cos(theta)
    large_surface_y_base = r2*np.sin(theta)
    return small_surface_x_base, small_surface_y_base, large_surface_x_base, \
                large_surface_y_base, r1, r2

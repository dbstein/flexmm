import numpy as np
import numba
import scipy as sp
import scipy.linalg
from .precomputations import Precomputation, Precomputations

"""
Define necessary functions and precomputations for KI-Style FMM
"""

def get_functions(functions):

    kernel_apply              = functions['kernel_apply']
    kernel_apply_single       = functions['kernel_apply_single']

    ############################################################################
    # These functions DEPEND on the particular FMM implementation

    @numba.njit(fastmath=True)
    def source_to_partial_multipole(sx, sy, tau, ucheck, cx, cy):
        kernel_apply(sx, sy, cx, cy, tau, ucheck)

    def partial_multipole_to_multipole(pM, LU):
        return sp.linalg.lu_solve(LU, pM.T, overwrite_b=True, check_finite=False).T

    def partial_local_to_local(pL, LU):
        return sp.linalg.lu_solve(LU, pL.T, overwrite_b=True, check_finite=False).T

    @numba.njit(fastmath=True)
    def local_expansion_to_target(expansion, tx, ty, sx, sy):
        return kernel_apply_single(sx, sy, tx, ty, expansion)

    new_functions = {
        'partial_multipole_to_multipole' : partial_multipole_to_multipole,
        'partial_local_to_local'         : partial_local_to_local,
        'source_to_partial_multipole'    : source_to_partial_multipole,
        'local_expansion_to_target'      : local_expansion_to_target,
    }
    functions.update(new_functions)

    # if gradient functions exist, wrap these up...
    if 'kernel_gradient_apply_single' in functions.keys():

        kernel_gradient_apply_single       = functions['kernel_gradient_apply_single']

        @numba.njit(fastmath=True)
        def local_expansion_gradient_to_target(expansion, tx, ty, sx, sy):
            return kernel_gradient_apply_single(sx, sy, tx, ty, expansion)

        new_functions = {
            'local_expansion_gradient_to_target' : local_expansion_gradient_to_target,
        }
        functions.update(new_functions)

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

class KI_Precomputations(Precomputations):
    def __init__(self, fmm, precomputations=None):
        super().__init__(precomputations)
        self.prepare(fmm)
    def prepare(self, fmm):
        tree = fmm.tree
        self.small_xs = []
        self.small_ys = []
        self.large_xs = []
        self.large_ys = []        
        for Level in tree.Levels:
            width = Level.width
            if not self.has_precomputation(width):
                precomp = KI_Precomputation(width, fmm)
                self.add_precomputation(precomp)
            precomp = self[width]
            self.small_xs.append(precomp.small_x)
            self.small_ys.append(precomp.small_y)
            self.large_xs.append(precomp.large_x)
            self.large_ys.append(precomp.large_y)
    def get_local_expansion_extras(self):
        return self.large_xs, self.large_ys

class KI_Precomputation(Precomputation):
    def __init__(self, width, fmm):
        super().__init__(width)
        self.precompute(fmm)
        self.compress()
    def precompute(self, fmm):
        width = self.width
        Nequiv = fmm.Nequiv
        KF = fmm.functions['kernel_form']
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
    def get_upwards_extras(self):
        return self.large_x, self.large_y
    def get_partial_multipole_to_multipole_extra(self):
        return self.S2L_LU
    def get_partial_local_to_local_extra(self):
        return self.L2S_LU

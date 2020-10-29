import numpy as np
import numba
import scipy as sp
import scipy.linalg

"""
Define necessary functions and precomputations for KI-Style FMM
"""

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
                precomputations['large_xs'][ind], precomputations['large_ys'][ind])

    def wrapped_local_expansion_evaluation(x, y, inds, locs, xmids, ymids, Local_Expansions, pot, precomputations):
        local_expansion_evaluation(x, y, inds, locs, xmids, ymids, Local_Expansions, pot, \
                precomputations['large_xs'], precomputations['large_ys'])

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
    # These functions DEPEND on the particular FMM implementation

    @numba.njit(fastmath=True)
    def source_to_partial_multipole(sx, sy, tau, ucheck, cx, cy):
        kernel_add(sx, sy, cx, cy, tau, ucheck)

    def partial_multipole_to_multipole(pM, precomputations, ind):
        return sp.linalg.lu_solve(precomputations['E2C_LUs'][ind], pM.T, overwrite_b=True, check_finite=False).T

    def partial_local_to_local(pL, precomputations, ind):
        return sp.linalg.lu_solve(precomputations['E2C_LUs'][ind], pL.T, overwrite_b=True, check_finite=False).T

    @numba.njit(fastmath=True)
    def local_expansion_to_target(expansion, tx, ty, sx, sy):
        return kernel_apply_single(sx, sy, tx, ty, expansion)

    ############################################################################
    # These functions DO NOT DEPEND on the particular FMM implementation

    new_functions = {
        'partial_multipole_to_multipole' : partial_multipole_to_multipole,
        'partial_local_to_local'         : partial_local_to_local,
        'source_to_partial_multipole'    : source_to_partial_multipole,
        'local_expansion_to_target'      : local_expansion_to_target,
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

def precompute(fmm, Nequiv):
    """
    Precomputations for KI-Style FMM
    kwargs:
        required:
            Nequiv: int, number of points used in equivalent surfaces
    """
    tree = fmm.tree
    Ncutoff = fmm.Ncutoff
    KF = fmm.functions['kernel_form']

    # generate the effective surfaces for each level
    theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
    small_xs = []
    small_ys = []
    large_xs = []
    large_ys = []
    small_radii = []
    large_radii = []
    widths = []
    for ind in range(tree.levels):
        Level = tree.Levels[ind]
        width = Level.width
        small_x, small_y, large_x, large_y, small_radius, large_radius = \
                                            get_level_information(width, theta, Nequiv)
        small_xs.append(small_x)
        small_ys.append(small_y)
        large_xs.append(large_x)
        large_ys.append(large_y)
        small_radii.append(small_radius)
        large_radii.append(large_radius)
        widths.append(width)
    # get C2E (check solution to equivalent density) operator for each level
    E2C_LUs = []
    for ind in range(tree.levels):
        equiv_to_check = Kernel_Form(KF, small_xs[ind], small_ys[ind], \
                                                large_xs[ind], large_ys[ind], mdtype=fmm.dtype)
        E2C_LUs.append(sp.linalg.lu_factor(equiv_to_check, overwrite_a=True, check_finite=False))
    # get Collected Equivalent Coordinates for each level
    M2MC = []
    for ind in range(tree.levels-1):
        collected_equiv_xs = np.concatenate([
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] - 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
                small_xs[ind+1] + 0.5*widths[ind+1],
            ])
        collected_equiv_ys = np.concatenate([
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
                small_ys[ind+1] - 0.5*widths[ind+1],
                small_ys[ind+1] + 0.5*widths[ind+1],
            ])
        Kern = Kernel_Form(KF, collected_equiv_xs, collected_equiv_ys, \
                                            large_xs[ind], large_ys[ind], mdtype=fmm.dtype)
        M2MC.append(Kern)
    # get L2LC operator
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
                    small_xhere = small_xs[ind] + (indx - 3)*widths[ind]
                    small_yhere = small_ys[ind] + (indy - 3)*widths[ind]
                    M2Lhere[indx,indy] = Kernel_Form(KF, small_xhere, \
                                            small_yhere, small_xs[ind], small_ys[ind], mdtype=fmm.dtype)
        M2LS.append(M2Lhere)

    precomputations = {
        'M2MC'     : M2MC,
        'L2LC'     : L2LC,
        'M2LS'     : M2LS,
        'large_xs' : large_xs,
        'large_ys' : large_ys,
        'E2C_LUs'  : E2C_LUs,
    }

    fmm.precomputations = precomputations

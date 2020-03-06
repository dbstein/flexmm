import numpy as np
import scipy as sp
import numexpr as ne
import scipy.linalg
import numba
import time
if __name__ != "__main__":
    from ..tree import Tree
    from ..float_dict import FloatDict
import sys

"""
Start out by writing a very specific Stokes "KI"-FMM
We will then go back and rewrite it to be more general, as in the scalar case
"""

def get_level_information(node_width, theta, N):
    # get information for this level
    dd = 0.1
    r1 = 0.5*node_width*(np.sqrt(2)+dd)
    r2 = 0.5*node_width*(4-np.sqrt(2)-2*dd)
    small_surface_x_base = r1*np.cos(theta)
    small_surface_y_base = r1*np.sin(theta)
    large_surface_x_base = r2*np.cos(theta)
    large_surface_y_base = r2*np.sin(theta)
    return small_surface_x_base, small_surface_y_base, large_surface_x_base, \
                large_surface_y_base, r1, r2
def get_normals(theta):
    normal_x = np.cos(theta)
    normal_y = np.sin(theta)
    return normal_x, normal_y

@numba.njit(parallel=True, fastmath=True)
def kernel_add(sx, sy, tx, ty, taux, tauy, outu, outv):
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for j in numba.prange(tx.size):
        u_j = 0.0
        v_j = 0.0
        for i in range(sx.size):
            dx = tx[j] - sx[i]
            dy = ty[j] - sy[i]
            d2 = dx*dx + dy*dy
            id2 = 1.0/d2
            logid = 0.5*np.log(id2)
            dxdyid2 = dx*dy*id2
            u_j += (logid + dx*dx*id2)*taux[i]
            u_j += dxdyid2*tauy[i]
            v_j += dxdyid2*taux[i]
            v_j += (logid + dy*dy*id2)*tauy[i]
        outu[j] = u_j*uscale
        outv[j] = v_j*uscale
    p0 = 0.0
    for i in range(sx.size):
        dx = tx[0] - sx[i]
        dy = ty[0] - sy[i]
        d2 = dx*dx + dy*dy
        id2 = 1.0/d2
        p0 += id2*(dx*taux[i]+dy*tauy[i])
    p0 *= pscale
    return p0

@numba.njit(parallel=True, fastmath=True)
def kernel_add_with_dipole(sx, sy, tx, ty, taux, tauy, sigmax, sigmay, dipx, dipy, outu, outv):
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for j in numba.prange(tx.size):
        u_j = 0.0
        v_j = 0.0
        for i in range(sx.size):
            dx = tx[j] - sx[i]
            dy = ty[j] - sy[i]
            d2 = dx*dx + dy*dy
            id2 = 1.0/d2
            logid = 0.5*np.log(id2)
            dxdyid2 = dx*dy*id2
            d_dot_n = dx*dipx[i] + dy*dipy[i]
            d_dot_n_ir4 = 4*d_dot_n*id2*id2
            Gd00 = d_dot_n_ir4*dx*dx
            Gd01 = d_dot_n_ir4*dx*dy
            Gd11 = d_dot_n_ir4*dy*dy
            u_j += (logid + dx*dx*id2)*taux[i] + Gd00*sigmax[i]
            u_j += dxdyid2*tauy[i] + Gd01*sigmay[i]
            v_j += dxdyid2*taux[i] + Gd01*sigmax[i]
            v_j += (logid + dy*dy*id2)*tauy[i] + Gd11*sigmay[i]
        outu[j] = u_j*uscale
        outv[j] = v_j*uscale
    p0 = 0.0
    for i in range(sx.size):
        dx = tx[0] - sx[i]
        dy = ty[0] - sy[i]
        d2 = dx*dx + dy*dy
        id2 = 1.0/d2
        d_dot_n = dx*dipx[i] + dy*dipy[i]
        d_dot_n_ir4 = d_dot_n*id2*id2
        p0 += id2*(dx*taux[i]+dy*tauy[i])
        p0 += 2*(-dipx[i]*id2 + 2*dx*d_dot_n_ir4)*sigmax[i]
        p0 += 2*(-dipy[i]*id2 + 2*dy*d_dot_n_ir4)*sigmay[i]
    p0 *= pscale
    return p0

@numba.njit(fastmath=True)
def kernel_apply_single_check(sx, sy, tx, ty, taux, tauy):
    u = 0.0
    v = 0.0
    p = 0.0
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for i in range(sx.size):
        dx = tx - sx[i]
        dy = ty - sy[i]
        if not (dx == 0 and dy == 0):
            d2 = dx*dx + dy*dy
            id2 = 1.0/d2
            logid = 0.5*np.log(id2)
            dxdyid2 = dx*dy*id2
            u += (logid + dx*dx*id2)*taux[i]
            u += dxdyid2*tauy[i]
            v += dxdyid2*taux[i]
            v += (logid + dy*dy*id2)*tauy[i]
            p += taux[i]*dx*id2
            p += tauy[i]*dy*id2
    u *= uscale
    v *= uscale
    p *= pscale
    return u, v, p

@numba.njit(fastmath=True)
def kernel_apply_single_check_with_dipole(sx, sy, tx, ty, taux, tauy, sigmax, sigmay, dipx, dipy):
    u = 0.0
    v = 0.0
    p = 0.0
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for i in range(sx.size):
        dx = tx - sx[i]
        dy = ty - sy[i]
        if not (dx == 0 and dy == 0):
            d2 = dx*dx + dy*dy
            id2 = 1.0/d2
            logid = 0.5*np.log(id2)
            dxdyid2 = dx*dy*id2
            d_dot_n = dx*dipx[i] + dy*dipy[i]
            d_dot_n_ir4 = 4*d_dot_n*id2*id2
            Gd00 = d_dot_n_ir4*dx*dx
            Gd01 = d_dot_n_ir4*dx*dy
            Gd11 = d_dot_n_ir4*dy*dy
            u += (logid + dx*dx*id2)*taux[i] + Gd00*sigmax[i]
            u += dxdyid2*tauy[i] + Gd01*sigmay[i]
            v += dxdyid2*taux[i] + Gd01*sigmax[i]
            v += (logid + dy*dy*id2)*tauy[i] + Gd11*sigmay[i]
            p += taux[i]*dx*id2
            p += tauy[i]*dy*id2
            p += 2*(-dipx[i]*id2 + 0.5*dx*d_dot_n_ir4)*sigmax[i]
            p += 2*(-dipy[i]*id2 + 0.5*dy*d_dot_n_ir4)*sigmay[i]
    u *= uscale
    v *= uscale
    p *= pscale
    return u, v, p

@numba.njit(fastmath=True)
def kernel_apply_single(sx, sy, tx, ty, taux, tauy):
    u = 0.0
    v = 0.0
    p = 0.0
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for i in range(sx.size):
        dx = tx - sx[i]
        dy = ty - sy[i]
        d2 = dx*dx + dy*dy
        id2 = 1.0/d2
        logid = 0.5*np.log(id2)
        dxdyid2 = dx*dy*id2
        u += (logid + dx*dx*id2)*taux[i]
        u += dxdyid2*tauy[i]
        v += dxdyid2*taux[i]
        v += (logid + dy*dy*id2)*tauy[i]
        p += taux[i]*dx*id2
        p += tauy[i]*dy*id2
    u *= uscale
    v *= uscale
    p *= pscale
    return u, v, p

@numba.njit(fastmath=True)
def kernel_apply_single_with_dipole(sx, sy, tx, ty, taux, tauy, sigmax, sigmay, dipx, dipy):
    u = 0.0
    v = 0.0
    p = 0.0
    uscale = 0.25/np.pi
    pscale = 2*uscale
    for i in range(sx.size):
        dx = tx - sx[i]
        dy = ty - sy[i]
        d2 = dx*dx + dy*dy
        id2 = 1.0/d2
        logid = 0.5*np.log(id2)
        dxdyid2 = dx*dy*id2
        d_dot_n = dx*dipx[i] + dy*dipy[i]
        d_dot_n_ir4 = 4*d_dot_n*id2*id2
        Gd00 = d_dot_n_ir4*dx*dx
        Gd01 = d_dot_n_ir4*dx*dy
        Gd11 = d_dot_n_ir4*dy*dy
        u += (logid + dx*dx*id2)*taux[i] + Gd00*sigmax[i]
        u += dxdyid2*tauy[i] + Gd01*sigmay[i]
        v += dxdyid2*taux[i] + Gd01*sigmax[i]
        v += (logid + dy*dy*id2)*tauy[i] + Gd11*sigmay[i]
        p += id2*(dx*taux[i]+dy*tauy[i])
        p += 2*(-dipx[i]*id2 + 0.5*dx*d_dot_n_ir4)*sigmax[i]
        p += 2*(-dipy[i]*id2 + 0.5*dy*d_dot_n_ir4)*sigmay[i]
        # dx = tx - sx[i]
        # dy = ty - sy[i]
        # d2 = dx*dx + dy*dy
        # id2 = 1.0/d2
        # logid = 0.5*np.log(id2)
        # dxdyid2 = dx*dy*id2
        # d_dot_n = dx*dipx[i] + dy*dipy[i]
        # d_dot_n_ir4 = 4*d_dot_n*id2*id2
        # Gd00 = d_dot_n_ir4*dx*dx
        # Gd01 = d_dot_n_ir4*dx*dy
        # Gd11 = d_dot_n_ir4*dy*dy
        # u += (logid + dx*dx*id2)*taux[i] + Gd00*sigmax[i]
        # u += dxdyid2*tauy[i] + Gd01*sigmay[i]
        # v += dxdyid2*taux[i] + Gd01*sigmax[i]
        # v += (logid + dy*dy*id2)*tauy[i] + Gd11*sigmay[i]
        # p += taux[i]*dx*id2
        # p += tauy[i]*dy*id2
        # p += 2*(-dipx[i]*id2 + 0.5*dx*d_dot_n_ir4)*sigmax[i]
        # p += 2*(-dipy[i]*id2 + 0.5*dy*d_dot_n_ir4)*sigmay[i]
    u *= uscale
    v *= uscale
    p *= pscale
    return u, v, p

@numba.njit(parallel=True)
def distribute(ucs, temp, pi, li, li2):
    for i in numba.prange(pi.size):
        if li[i] >= 0:
            ucs[li2[pi[i]]] = temp[li[i]]

@numba.njit(parallel=True)
def add_interactions_prepared(M2Ls, PLEs, dilist, Nequiv):
    # loop over leaves in this level
    for i in numba.prange(dilist.shape[0]):
        di = dilist[i]
        if di >= 0:
            for k in range(2*Nequiv+1):
                PLEs[i,k] += M2Ls[di,k]

def fake_print(*args, **kwargs):
    pass
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

@numba.njit(fastmath=True)
def local_expansion_to_target(expansion_x, expansion_y, tx, ty, sx, sy):
    return kernel_apply_single(sx, sy, tx, ty, expansion_x, expansion_y)

@numba.njit(parallel=True, fastmath=True)
def upwards_pass(x, y, li, cu, bind, tind, xmid, ymid, taux, tauy, pM, e1, e2):
    """
    Generic upwards pass function
    ns: number of all sources
    nL: number of leaves in this level
    nl: number of leaves where multipole expansions need to be calculated
    x,    f8[ns]   - array of all source x values (ordered)
    y,    f8[ns]   - array of all source y values (ordered)
    li,   i8[nL]   - link indeces, see explanation in tree
    cu,   b[nL]    - whether an upwards computation needs to be done
    bind, i8[nL]   - lower source index for leaf
    tind, i8[nL]   - upper source index for leaf
    xmid, f8[nL]   - midpoint of leaf at level (x-coord)
    ymid, f8[nL]   - midpoint of leaf at level (y-coord)
    tau,  *[ns]    - density (ordered)
                        * is f8 or c16
    pM,   *[nl,**] - partial multipole expansion
                        * is f8 or c16
                        ** is deferred to the underlying FMM method
    e1,   *[**]    - extra array used by method
    e2,   *[**]    - extra array used by method
    """
    Nequiv = e1.size
    for i in numba.prange(bind.size):
        if cu[i] and li[i] >= 0:
            bi = bind[i]
            ti = tind[i]
            kernel_add(x[bi:ti]-xmid[i], y[bi:ti]-ymid[i], e1, e2,
                taux[bi:ti], tauy[bi:ti], pM[li[i], 0*Nequiv:1*Nequiv], pM[li[i], 1*Nequiv:2*Nequiv])

@numba.njit(parallel=True, fastmath=True)
def upwards_pass_dipole(x, y, li, cu, bind, tind, xmid, ymid, taux, tauy, sigmax, sigmay, dipx, dipy, pM, e1, e2):
    """
    Generic upwards pass function
    ns: number of all sources
    nL: number of leaves in this level
    nl: number of leaves where multipole expansions need to be calculated
    x,    f8[ns]   - array of all source x values (ordered)
    y,    f8[ns]   - array of all source y values (ordered)
    li,   i8[nL]   - link indeces, see explanation in tree
    cu,   b[nL]    - whether an upwards computation needs to be done
    bind, i8[nL]   - lower source index for leaf
    tind, i8[nL]   - upper source index for leaf
    xmid, f8[nL]   - midpoint of leaf at level (x-coord)
    ymid, f8[nL]   - midpoint of leaf at level (y-coord)
    tau,  *[ns]    - density (ordered)
                        * is f8 or c16
    pM,   *[nl,**] - partial multipole expansion
                        * is f8 or c16
                        ** is deferred to the underlying FMM method
    e1,   *[**]    - extra array used by method
    e2,   *[**]    - extra array used by method
    """
    Nequiv = e1.size
    for i in numba.prange(bind.size):
        if cu[i] and li[i] >= 0:
            bi = bind[i]
            ti = tind[i]
            kernel_add_with_dipole(x[bi:ti]-xmid[i], y[bi:ti]-ymid[i], e1, e2,
                taux[bi:ti], tauy[bi:ti], sigmax[bi:ti], sigmay[bi:ti], dipx[bi:ti], dipy[bi:ti], pM[li[i], 0*Nequiv:1*Nequiv], pM[li[i], 1*Nequiv:2*Nequiv])

@numba.njit(parallel=True, fastmath=True)
def local_expansion_evaluation(tx, ty, inds, locs, xmids, ymids, LEs, potu, potv, potp, e1, e2):
    """
    Generic local expansion evalution
    nt: number of targets
    nL: number of levels
    tx,    f8[nt]   - array of all target x values
    ty,    f8[nt]   - array of all target y values
    inds,  i8[nt]   - which level this target is in
    locs,  i8[nt]   - location in level information for this target
    xmids, list(nL) - list of xmids for whole tree
                    - each element is a f8[# leaves in level]
    ymids, list(nL) - list of ymids for whole tree
    LEs,   list(nL) - local expansions for whole tree
    pot,   *[nt]    - potential
    e1,    list(nL) - list of extra things specific to method
    e2,    list(nL) - list of extra things specific to method
    """
    for i in numba.prange(tx.size):
        x = tx[i]
        y = ty[i]
        ind = inds[i]
        loc = locs[i]
        x = x - xmids[ind][loc]
        y = y - ymids[ind][loc]
        out = local_expansion_to_target(LEs[ind][loc,0], LEs[ind][loc,1], x, y, e1[ind], e2[ind])
        potu[i] = out[0]
        potv[i] = out[1]
        potp[i] = out[2]

@numba.njit(parallel=True, fastmath=True)
def neighbor_evaluation(tx, ty, sx, sy, inds, locs, binds, tinds, colls, tauox, tauoy, potu, potv, potp, check):
    """
    Generic neighbor evalution
    nt: number of targets
    ns: number of sources
    nL: number of levels
    tx,    f8[nt]   - array of all target x values
    ty,    f8[nt]   - array of all target y values
    sx,    f8[ns]   - array of all source x values (ordered)
    sy,    f8[ns]   - array of all source y values (ordered)
    inds,  i8[nt]   - which level this target is in
    locs,  i8[nt]   - location in level information for this target
    binds, list[nL] - list of all lower indeces into source information
    tinds, list[nL] - list of all upper indeces into source information
    colls, list[nL] - list of all colleagues
    tauo,  *[ns]    - density, ordered
    pot,   *[nt]    - potential
    check, bool     - whether to check for source/targ coincidences
    """
    for i in numba.prange(tx.size):
        x = tx[i]
        y = ty[i]
        ind = inds[i]
        loc = locs[i]
        cols = colls[ind][loc]
        for j in range(9):
            ci = cols[j]
            if ci >= 0:
                bind = binds[ind][ci]
                tind = tinds[ind][ci]
                if tind - bind > 0:
                    if check:
                        out = kernel_apply_single_check(sx[bind:tind], sy[bind:tind], x, y, tauox[bind:tind], tauoy[bind:tind])
                        potu[i] += out[0]
                        potv[i] += out[1]
                        potp[i] += out[2]
                    else:
                        out = kernel_apply_single(sx[bind:tind], sy[bind:tind], x, y, tauox[bind:tind], tauoy[bind:tind])
                        potu[i] += out[0]
                        potv[i] += out[1]
                        potp[i] += out[2]


@numba.njit(parallel=True, fastmath=True)
def neighbor_evaluation_dipole(tx, ty, sx, sy, inds, locs, binds, tinds, colls, tauox, tauoy, sigmaox, sigmaoy, dipox, dipoy, potu, potv, potp, check):
    """
    Generic neighbor evalution
    nt: number of targets
    ns: number of sources
    nL: number of levels
    tx,    f8[nt]   - array of all target x values
    ty,    f8[nt]   - array of all target y values
    sx,    f8[ns]   - array of all source x values (ordered)
    sy,    f8[ns]   - array of all source y values (ordered)
    inds,  i8[nt]   - which level this target is in
    locs,  i8[nt]   - location in level information for this target
    binds, list[nL] - list of all lower indeces into source information
    tinds, list[nL] - list of all upper indeces into source information
    colls, list[nL] - list of all colleagues
    tauo,  *[ns]    - density, ordered
    pot,   *[nt]    - potential
    check, bool     - whether to check for source/targ coincidences
    """
    for i in numba.prange(tx.size):
        x = tx[i]
        y = ty[i]
        ind = inds[i]
        loc = locs[i]
        cols = colls[ind][loc]
        for j in range(9):
            ci = cols[j]
            if ci >= 0:
                bind = binds[ind][ci]
                tind = tinds[ind][ci]
                if tind - bind > 0:
                    if check:
                        out = kernel_apply_single_check_with_dipole(sx[bind:tind], sy[bind:tind], x, y, tauox[bind:tind], tauoy[bind:tind], sigmaox[bind:tind], sigmaoy[bind:tind], dipox[bind:tind], dipoy[bind:tind])
                        potu[i] += out[0]
                        potv[i] += out[1]
                        potp[i] += out[2]
                    else:
                        out = kernel_apply_single_with_dipole(sx[bind:tind], sy[bind:tind], x, y, tauox[bind:tind], tauoy[bind:tind], sigmaox[bind:tind], sigmaoy[bind:tind], dipox[bind:tind], dipoy[bind:tind])
                        potu[i] += out[0]
                        potv[i] += out[1]
                        potp[i] += out[2]

class M2L_Evaluator(object):
    def __init__(self, m2l, svd_tol):
        self.m2l = m2l
        self.svd_tol = svd_tol
        S = np.linalg.svd(self.m2l)
        cuts = np.where(S[1] > svd_tol)[0]
        if len(cuts) > 0:
            cutoff = np.where(S[1] > svd_tol)[0][-1] + 1
            # if this is the case, svd compression probably saves time
            if cutoff < m2l.shape[0]:
            # if False: # turn this off for now
                self.M2 = S[0][:,:cutoff].copy()
                A1 = S[1][:cutoff]
                A2 = S[2][:cutoff]
                self.M1 = A2*A1[:,None]
                self._call = self._call_svd
                self.cutoff = cutoff
            else:
                self._call = self._call_mat
        else:
            self._call = self._call_null
    def _call_mat(self, tau, out, work):
        np.matmul(self.m2l, tau, out)
    def _call_svd(self, tau, out, work):
        np.matmul(self.M1, tau, out=work)
        np.matmul(self.M2, work, out=out)
    def _call_null(self, tau, out, work):
        out[:] = 0.0
    def __call__(self, tau, out, work=None):
        self._call(tau, out, work)

def Stokes_Kernel_Form(sx, sy, tx, ty):
    ns = sx.size
    nt = tx.size
    SX = sx
    SY = sy
    TX = tx[:,None]
    TY = ty[:,None]
    fscale = 0.25/np.pi
    pscale = 2*fscale
    G = np.zeros([2*nt+1, 2*ns], dtype=float)
    dx = ne.evaluate('TX - SX')
    dy = ne.evaluate('TY - SY')
    id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
    W = np.empty_like(dx)
    GH = np.empty_like(dx)
    # forces effect on velocity
    logid = ne.evaluate('0.5*log(id2)', out=W)
    ne.evaluate('fscale*(logid + dx*dx*id2)', out=GH)
    G[0*nt:1*nt, 0*ns:1*ns] += GH
    ne.evaluate('fscale*dx*dy*id2', out=GH)
    G[1*nt:2*nt, 0*ns:1*ns] += GH
    G[0*nt:1*nt, 1*ns:2*ns] += GH            
    GH = ne.evaluate('fscale*(logid + dy*dy*id2)')
    G[1*nt:2*nt, 1*ns:2*ns] += GH
    # add pressure computation
    txmean = np.mean(tx)
    tymean = np.mean(ty)
    dx = txmean-sx
    dy = tymean-sy
    id2 = 1.0/(dx**2 + dy**2)
    G[-1, :] = pscale*np.concatenate([dx*id2, dy*id2])
    return G

def Corrected_Stokes_Kernel_Form(sx, sy, tx, ty):
    ns = sx.size
    nt = tx.size
    SX = sx
    SY = sy
    TX = tx[:,None]
    TY = ty[:,None]
    fscale = 0.25/np.pi
    pscale = 2*fscale
    G = np.zeros([2*nt+1, 2*ns+1], dtype=float)
    dx = ne.evaluate('TX - SX')
    dy = ne.evaluate('TY - SY')
    id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
    W = np.empty_like(dx)
    GH = np.empty_like(dx)
    # forces effect on velocity
    logid = ne.evaluate('0.5*log(id2)', out=W)
    ne.evaluate('fscale*(logid + dx*dx*id2)', out=GH)
    G[0*nt:1*nt, 0*ns:1*ns] += GH
    ne.evaluate('fscale*dx*dy*id2', out=GH)
    G[1*nt:2*nt, 0*ns:1*ns] += GH
    G[0*nt:1*nt, 1*ns:2*ns] += GH            
    GH = ne.evaluate('fscale*(logid + dy*dy*id2)')
    G[1*nt:2*nt, 1*ns:2*ns] += GH
    # add pressure computation and correction
    txmean = np.mean(tx)
    tymean = np.mean(ty)
    dx = txmean-sx
    dy = tymean-sy
    id2 = 1.0/(dx**2 + dy**2)
    G[-1, :-1] = pscale*np.concatenate([dx*id2, dy*id2])
    G[:-1, -1] = G[-1, :-1]
    return G

class SVD_Solver(object):
    def __init__(self, A, tol=1e-15):
        self.A = A
        self.U, S, self.VH = np.linalg.svd(self.A)
        S[S < tol] = np.Inf
        self.SI = 1.0/S
    def __call__(self, b):
        mult = self.SI[:,None] if len(b.shape) > 1 else self.SI
        return self.VH.T.dot(mult*self.U.T.dot(b))

class Precomputation(object):
    def __init__(self, width, Nequiv):
        self.Nequiv = Nequiv
        self.width = width
        theta = np.linspace(0, 2*np.pi, Nequiv, endpoint=False)
        small_x, small_y, large_x, large_y, small_radius, large_radius = \
                                    get_level_information(width, theta, Nequiv)
        self.small_x = small_x
        self.small_y = small_y
        self.large_x = large_x
        self.large_y = large_y
        equiv_to_check = Corrected_Stokes_Kernel_Form(small_x, small_y, large_x, large_y) 
        self.E2C_SVD = SVD_Solver(equiv_to_check)
        check_to_equiv = Corrected_Stokes_Kernel_Form(large_x, large_y, small_x, small_y)
        self.C2E_SVD = SVD_Solver(check_to_equiv)
        # get Collected Equivalent Coordinates for each level
        ssx, ssy, _, _, _, _ = get_level_information(0.5*width, theta, Nequiv)
        Kern1 = Stokes_Kernel_Form(ssx - 0.25*width, ssy - 0.25*width, large_x, large_y)
        Kern2 = Stokes_Kernel_Form(ssx - 0.25*width, ssy + 0.25*width, large_x, large_y)
        Kern3 = Stokes_Kernel_Form(ssx + 0.25*width, ssy - 0.25*width, large_x, large_y)
        Kern4 = Stokes_Kernel_Form(ssx + 0.25*width, ssy + 0.25*width, large_x, large_y)
        self.M2MC = np.column_stack([Kern1, Kern2, Kern3, Kern4])
        Kern1 = Stokes_Kernel_Form(large_x, large_y, ssx - 0.25*width, ssy - 0.25*width)
        Kern2 = Stokes_Kernel_Form(large_x, large_y, ssx - 0.25*width, ssy + 0.25*width)
        Kern3 = Stokes_Kernel_Form(large_x, large_y, ssx + 0.25*width, ssy - 0.25*width)
        Kern4 = Stokes_Kernel_Form(large_x, large_y, ssx + 0.25*width, ssy + 0.25*width)
        self.L2LC = np.row_stack([Kern1, Kern2, Kern3, Kern4])
        M2Ls = np.empty([7,7], dtype=object)
        M2LF = np.empty([7,7], dtype=object)
        for indx in range(7):
            for indy in range(7):
                if indx-3 in [-1, 0, 1] and indy-3 in [-1, 0, 1]:
                    M2Ls[indx, indy] = None
                    M2LF[indx, indy] = None
                else:
                    small_xhere = small_x + (indx - 3)*width
                    small_yhere = small_y + (indy - 3)*width
                    M2Ls[indx,indy] = Stokes_Kernel_Form(small_xhere, small_yhere, small_x, small_y)
                    M2LF[indx,indy] = M2L_Evaluator(M2Ls[indx,indy], 1e-14)
        largest_cutoff = 0
        for i in range(7):
            for j in range(7):
                if hasattr(M2LF[i,j], 'cutoff'):
                    largest_cutoff = max(largest_cutoff, M2LF[i,j].cutoff)
        self.M2Ls = M2Ls
        self.M2LF = M2LF
        self.largest_cutoff = largest_cutoff

class FMM(object):
    def __init__(self, x, y, Nequiv=48, Ncutoff=50, bbox=None, precomputations=None, verbose=False):
        self.x = x
        self.y = y
        self.bbox = bbox
        self.Nequiv = Nequiv
        self.Ncutoff = Ncutoff
        self.input_precomputations = precomputations
        self.verbose = verbose
        self.print = get_print_function(self.verbose)
        if self.input_precomputations is not None:
            self.adjust_bbox()
        self.build_tree()
    def adjust_bbox(self):
        """
        make sure the bbox is compatible with our precomptuations...
        """
        px, py = self.x, self.y
        bbox = self.bbox
        precomputations = self.input_precomputations
        if bbox is None:
            bbox = [np.min(px), np.max(px), np.min(py), np.max(py)]
        if precomputations is not None:
            # get the smallest width we know about
            widths = np.array([width for width in precomputations])
            if len(widths) > 0:
                small_width = np.min(widths)
                # get width required by bbox
                bbox_x_width = bbox[1] - bbox[0]
                bbox_y_width = bbox[3] - bbox[2]
                bbox_width = max(bbox_x_width, bbox_y_width)
                # double small_width until we're bigger than bbox_width
                width = 2**np.ceil(np.log2(bbox_width/small_width))*small_width
            else:
                width = max(bbox[1]-bbox[0], bbox[3]-bbox[2])
            # adjust bbox so that we can reuse precomputations
            bbox_x_center = (bbox[1] + bbox[0])/2
            bbox_y_center = (bbox[3] + bbox[2])/2
            bbox_x = [bbox_x_center - width/2, bbox_x_center + width/2]
            bbox_y = [bbox_y_center - width/2, bbox_y_center + width/2]
            self.bbox = bbox_x + bbox_y
        else:
            bbox = self.bbox
    def build_tree(self):
        st = time.time()
        self.tree = Tree(self.x, self.y, self.Ncutoff, self.bbox)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
    def precompute(self):
        tree = self.tree
        self.precomputations = FloatDict()
        if self.input_precomputations is not None:
            self.precomputations.update(self.input_precomputations)
        self.large_xs = []
        self.large_ys = []
        for Level in tree.Levels:
            if Level.width not in self.precomputations:
                prec = Precomputation(Level.width, self.Nequiv)
                self.precomputations[Level.width] = prec
            prec = self.precomputations[Level.width]
            self.large_xs.append(prec.large_x)
            self.large_ys.append(prec.large_y)
    def build_expansions(self, taux, tauy):
        tree = self.tree
        precomputations = self.precomputations

        Nequiv, Ncutoff = self.Nequiv, self.Ncutoff
        
        taux_ordered = taux[tree.ordv]
        tauy_ordered = tauy[tree.ordv]
        self.taux = taux
        self.tauy = tauy
        self.taux_ordered = taux_ordered
        self.tauy_ordered = tauy_ordered
        self.multipoles = [None,]*tree.levels
        self.reshaped_multipoles = [None,]*tree.levels
        # upwards pass - start at bottom leaf nodes and build multipoles up
        st = time.time()
        for ind in reversed(range(tree.levels)[2:]):
            Level = tree.Levels[ind]
            prec = self.precomputations[Level.width]
            # allocate space for the partial multipoles
            anum = Level.n_allocate_density
            partial_multipole = np.zeros([anum, 2*Nequiv+1], dtype=float)
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = prec.M2MC.dot(self.reshaped_multipoles[ind+1].T).T.copy()
                distribute(partial_multipole, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            upwards_pass(tree.x, tree.y, Level.this_density_ind, Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid, taux_ordered, tauy_ordered, partial_multipole, prec.large_x, prec.large_y)
            self.multipoles[ind] = prec.E2C_SVD(partial_multipole.T).T.copy()
            resh = (int(anum/4), Nequiv*8)
            self.reshaped_multipoles[ind] = np.reshape(self.multipoles[ind][:,:-1], resh)
            if self.reshaped_multipoles[ind].flags.owndata:
                raise Exception('Something went wrong with reshaping the equivalent densities, it made a copy instead of a view.')
        et = time.time()
        self.print('....Time for upwards pass:      {:0.2f}'.format(1000*(et-st)))
        # downwards pass - start at top and work down to build up local expansions
        st = time.time()
        self.Partial_Local_Expansions = [None,]*tree.levels
        self.resh_Local_Expansions = [None,]*tree.levels
        self.Local_Expansions = [None,]*tree.levels
        self.Local_Expansions[0] = np.zeros([1, 2*Nequiv], dtype=float)
        self.resh_Local_Expansions[0] = self.Local_Expansions[0].reshape(1, 2, Nequiv)
        try:
            self.Local_Expansions[1] = np.zeros([4, 2*Nequiv], dtype=float)
            self.resh_Local_Expansions[1] = self.Local_Expansions[1].reshape(4, 2, Nequiv)
        except:
            pass
        for ind in range(2, tree.levels):
            Level = tree.Levels[ind]
            prec = self.precomputations[Level.width]
            Parent_Level = tree.Levels[ind-1]
            if ind == 2:
                self.Partial_Local_Expansions[ind] = np.zeros([16, 2*Nequiv+1], dtype=float)
            M2Ls = np.zeros([self.multipoles[ind].shape[0], 2*Nequiv+1])
            # build the interaction lists
            dilists = tree.interaction_lists[ind]
            # add up interactions
            work = np.empty([prec.largest_cutoff, self.multipoles[ind].shape[0]], dtype=float)
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        workmat = work[:prec.M2LF[i,j].cutoff] if hasattr(prec.M2LF[i,j], 'cutoff') else None
                        prec.M2LF[i,j](self.multipoles[ind][:,:-1].T, out=M2Ls.T, work=workmat)
                        add_interactions_prepared(M2Ls, self.Partial_Local_Expansions[ind], dilists[i,j], Nequiv)
            # convert partial local expansions to local local_expansions
            self.Local_Expansions[ind] = prec.C2E_SVD(self.Partial_Local_Expansions[ind].T).T.copy()[:,:-1]
            self.resh_Local_Expansions[ind] = self.Local_Expansions[ind].reshape(self.Local_Expansions[ind].shape[0], 2, Nequiv).copy()
            # move local expansions downwards
            if ind < tree.levels-1:
                doit = Level.compute_downwards
                descendant_level = tree.Levels[ind+1]
                local_expansions = self.Local_Expansions[ind][doit]
                partial_local_expansions = prec.L2LC.dot(local_expansions.T).T
                sorter = np.argsort(Level.children_ind[doit])
                self.Partial_Local_Expansions[ind+1] = partial_local_expansions[sorter].reshape([descendant_level.n_node, 2*Nequiv+1])
        et = time.time()
        self.print('....Time for downwards pass:    {:0.2f}'.format(1000*(et-st)))

    def build_expansions_dipole(self, taux, tauy, sigmax, sigmay, dipx, dipy):
        tree = self.tree
        precomputations = self.precomputations

        Nequiv, Ncutoff = self.Nequiv, self.Ncutoff
        
        taux_ordered = taux[tree.ordv]
        tauy_ordered = tauy[tree.ordv]
        sigmax_ordered = sigmax[tree.ordv]
        sigmay_ordered = sigmay[tree.ordv]
        dipx_ordered = dipx[tree.ordv]
        dipy_ordered = dipy[tree.ordv]
        self.taux = taux
        self.tauy = tauy
        self.taux_ordered = taux_ordered
        self.tauy_ordered = tauy_ordered
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.sigmax_ordered = sigmax_ordered
        self.sigmay_ordered = sigmay_ordered
        self.dipx = dipx
        self.dipy = dipy
        self.dipx_ordered = dipx_ordered
        self.dipy_ordered = dipy_ordered
        self.multipoles = [None,]*tree.levels
        self.reshaped_multipoles = [None,]*tree.levels
        # upwards pass - start at bottom leaf nodes and build multipoles up
        st = time.time()
        for ind in reversed(range(tree.levels)[2:]):
            Level = tree.Levels[ind]
            prec = precomputations[Level.width]
            # allocate space for the partial multipoles
            anum = Level.n_allocate_density
            partial_multipole = np.zeros([anum, 2*Nequiv+1], dtype=float)
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = prec.M2MC.dot(self.reshaped_multipoles[ind+1].T).T.copy()
                distribute(partial_multipole, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            upwards_pass_dipole(tree.x, tree.y, Level.this_density_ind, Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid, taux_ordered, tauy_ordered, sigmax_ordered, sigmay_ordered, dipx_ordered, dipy_ordered, partial_multipole, prec.large_x, prec.large_y)
            self.multipoles[ind] = prec.E2C_SVD(partial_multipole.T).T.copy()
            resh = (int(anum/4), Nequiv*8)
            self.reshaped_multipoles[ind] = np.reshape(self.multipoles[ind][:,:-1], resh)
            if self.reshaped_multipoles[ind].flags.owndata:
                raise Exception('Something went wrong with reshaping the equivalent densities, it made a copy instead of a view.')
        et = time.time()
        self.print('....Time for upwards pass:      {:0.2f}'.format(1000*(et-st)))
        # downwards pass - start at top and work down to build up local expansions
        st = time.time()
        self.Partial_Local_Expansions = [None,]*tree.levels
        self.resh_Local_Expansions = [None,]*tree.levels
        self.Local_Expansions = [None,]*tree.levels
        self.Local_Expansions[0] = np.zeros([1, 2*Nequiv], dtype=float)
        self.resh_Local_Expansions[0] = self.Local_Expansions[0].reshape(1, 2, Nequiv)
        try:
            self.Local_Expansions[1] = np.zeros([4, 2*Nequiv], dtype=float)
            self.resh_Local_Expansions[1] = self.Local_Expansions[1].reshape(4, 2, Nequiv)
        except:
            pass
        for ind in range(2, tree.levels):
            Level = tree.Levels[ind]
            prec = precomputations[Level.width]
            Parent_Level = tree.Levels[ind-1]
            if ind == 2:
                self.Partial_Local_Expansions[ind] = np.zeros([16, 2*Nequiv+1], dtype=float)
            M2Ls = np.zeros([self.multipoles[ind].shape[0], 2*Nequiv+1])
            # build the interaction lists
            dilists = tree.interaction_lists[ind]
            # add up interactions
            work = np.empty([prec.largest_cutoff, self.multipoles[ind].shape[0]], dtype=float)
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        workmat = work[:prec.M2LF[i,j].cutoff] if hasattr(prec.M2LF[i,j], 'cutoff') else None
                        prec.M2LF[i,j](self.multipoles[ind][:,:-1].T, out=M2Ls.T, work=workmat)
                        add_interactions_prepared(M2Ls, self.Partial_Local_Expansions[ind], dilists[i,j], Nequiv)
            # convert partial local expansions to local local_expansions
            self.Local_Expansions[ind] = prec.C2E_SVD(self.Partial_Local_Expansions[ind].T).T.copy()[:,:-1]
            self.resh_Local_Expansions[ind] = self.Local_Expansions[ind].reshape(self.Local_Expansions[ind].shape[0], 2, Nequiv).copy()
            # move local expansions downwards
            if ind < tree.levels-1:
                doit = Level.compute_downwards
                descendant_level = tree.Levels[ind+1]
                local_expansions = self.Local_Expansions[ind][doit]
                partial_local_expansions = prec.L2LC.dot(local_expansions.T).T
                sorter = np.argsort(Level.children_ind[doit])
                self.Partial_Local_Expansions[ind+1] = partial_local_expansions[sorter].reshape([descendant_level.n_node, 2*Nequiv+1])
        et = time.time()
        self.print('....Time for downwards pass:    {:0.2f}'.format(1000*(et-st)))

    def evaluate_to_points(self, x,  y, check_self=False):
        precomputations = self.precomputations
        tree = self.tree

        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate local expansions
        potu = np.zeros(x.size, dtype=float)
        potv = np.zeros(x.size, dtype=float)
        potp = np.zeros(x.size, dtype=float)
        Local_Expansions = self.resh_Local_Expansions
        local_expansion_evaluation(x, y, inds, locs, tree.xmids, tree.ymids, Local_Expansions, potu, potv, potp, self.large_xs, self.large_ys)
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.taux_ordered, self.tauy_ordered, potu, potv, potp, check_self)
        return potu, potv, potp

    def evaluate_to_points_dipole(self, x,  y, check_self=False):
        precomputations = self.precomputations
        tree = self.tree

        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate local expansions
        potu = np.zeros(x.size, dtype=float)
        potv = np.zeros(x.size, dtype=float)
        potp = np.zeros(x.size, dtype=float)
        Local_Expansions = self.resh_Local_Expansions
        local_expansion_evaluation(x, y, inds, locs, tree.xmids, tree.ymids, Local_Expansions, potu, potv, potp, self.large_xs, self.large_ys)
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation_dipole(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.taux_ordered, self.tauy_ordered, self.sigmax_ordered, self.sigmay_ordered, self.dipx_ordered, self.dipy_ordered, potu, potv, potp, check_self)
        return potu, potv, potp




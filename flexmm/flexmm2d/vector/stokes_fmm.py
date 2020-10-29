import numpy as np
import scipy as sp
import numexpr as ne
import scipy.linalg
import numba
import time
if __name__ != "__main__":
    from ..tree import Tree
import sys

"""
Start out by writing a very specific Stokes "KI"-FMM
We will then go back and rewrite it to be more general, as in the scalar case
"""

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
def get_normals(theta):
    normal_x = np.cos(theta)
    normal_y = np.sin(theta)
    return normal_x, normal_y

# This doesn't actually appear to be kernel_add?
# appears to be just kernel_apply...
@numba.njit(parallel=True, fastmath=True)
def kernel_add(sx, sy, tx, ty, taux, tauy, outu, outv):
    uscale = 0.25/np.pi
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

@numba.njit(parallel=True)
def distribute(ucs, temp, pi, li, li2):
    for i in numba.prange(pi.size):
        if li[i] >= 0:
            ucs[li2[pi[i]]] = temp[li[i]]

@numba.njit(parallel=True)
def build_interaction_list(parents, pcoll, pchild, xmid, ymid, width, li, dilists):
    """
    Ahead of time interaction list preparation
    This should probably be moved to the tree file
    """
    # loop over leaves in this level
    n = parents.size
    dilists[:] = -1
    for i in numba.prange(n):
        # parents index
        ind = parents[i]
        xmidi = xmid[i]
        ymidi = ymid[i]
        # loop over parents colleagues
        for j in range(9):
            pi = pcoll[ind, j]
            # if colleague exists and isn't the parent itself
            if pi >= 0 and pi != ind:
                pch = pchild[pi]
                # loop over colleagues children
                for k in range(4):
                    ci = pch + k
                    # get the distance offsets
                    xdist = xmid[ci]-xmidi
                    ydist = ymid[ci]-ymidi
                    xd = int(np.round(xdist/width))
                    yd = int(np.round(ydist/width))
                    # get index into mutlipoles 
                    di = li[ci]
                    # if the multipole was formed, add in interaction
                    if di >= 0:
                        for ii in range(7):
                            for jj in range(7):
                                if not (ii in [2,3,4] and jj in [2,3,4]):
                                    if xd == ii-3 and yd == jj-3:
                                        dilists[ii, jj, i] = di

@numba.njit(parallel=True)
def add_interactions_prepared(M2Ls, PLEs, dilist, Nequiv):
    # loop over leaves in this level
    for i in numba.prange(dilist.shape[0]):
        di = dilist[i]
        if di >= 0:
            for k in range(2*Nequiv):
                PLEs[i,k] += M2Ls[di,k]

def fake_print(*args, **kwargs):
    pass
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

def partial_multipole_to_multipole(pM, precomputations, ind):
    return (precomputations['E2C_SVDs'][ind](pM.T).T).copy()

def partial_local_to_local(pL, precomputations, ind):
    return (precomputations['C2E_SVDs'][ind](pL.T).T).copy()

@numba.njit(fastmath=True)
def source_to_partial_multipole(sx, sy, taux, tauy, ucheck, vcheck, cx, cy):
    kernel_add(sx, sy, cx, cy, taux, tauy, ucheck, vcheck)

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
    for i in numba.prange(bind.size):
        if cu[i] and li[i] >= 0:
            bi = bind[i]
            ti = tind[i]
            source_to_partial_multipole(x[bi:ti]-xmid[i],
                y[bi:ti]-ymid[i], taux[bi:ti], tauy[bi:ti], pM[li[i], 0], pM[li[i], 1], e1, e2)

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
    G = np.zeros([2*nt, 2*ns], dtype=float)
    dx = ne.evaluate('TX - SX')
    dy = ne.evaluate('TY - SY')
    id2 = ne.evaluate('1.0/(dx**2 + dy**2)')
    W = np.empty_like(dx)
    GH = np.empty_like(dx)
    # forces effect on velocity
    logid = ne.evaluate('0.5*log(id2)', out=W)
    ne.evaluate('fscale*(logid + dx*dx*id2)', out=GH)
    G[:nt, :ns] += GH
    ne.evaluate('fscale*dx*dy*id2', out=GH)
    G[nt:, :ns] += GH
    G[:nt, ns:] += GH            
    GH = ne.evaluate('fscale*(logid + dy*dy*id2)')
    G[nt:, ns:] += GH
    return G

def Add_Pressure_Fix(G, snx, sny, tnx, tny):
    SNX = snx
    SNY = sny
    TNX = tnx[:,None]
    TNY = tny[:,None]
    ns = snx.size
    nt = tnx.size
    # G[:,0] += np.concatenate([tnx, tny])
    W = G[:nt, :ns]
    ne.evaluate('W+TNX*SNX', out=W)
    W = G[nt:, :ns]
    ne.evaluate('W+TNY*SNX', out=W)
    W = G[:nt, ns:]
    ne.evaluate('W+TNX*SNY', out=W)
    W = G[nt:, ns:]
    ne.evaluate('W+TNY*SNY', out=W)

def Kernel_Form(sx, sy, tx, ty):
    return Stokes_Kernel_Form(sx, sy, tx, ty)

class SVD_Solver(object):
    def __init__(self, A, tol=1e-15):
        self.A = A
        self.U, S, self.VH = np.linalg.svd(self.A)
        S[S < tol] = np.Inf
        self.SI = 1.0/S
    def __call__(self, b):
        mult = self.SI[:,None] if len(b.shape) > 1 else self.SI
        return self.VH.T.dot(mult*self.U.T.dot(b))

class FMM(object):
    def __init__(self, x, y, Nequiv=48, Ncutoff=50, bbox=None, verbose=False):
        self.x = x
        self.y = y
        self.bbox = bbox
        self.Nequiv = Nequiv
        self.Ncutoff = Ncutoff
        self.verbose = verbose
        self.print = get_print_function(self.verbose)
        self.build_tree()
    def build_tree(self):
        st = time.time()
        self.tree = Tree(self.x, self.y, self.Ncutoff, self.bbox)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
    def precompute(self):
        """
        Precomputations for Stokes KI-Style FMM
        """
        tree = self.tree
        Ncutoff = self.Ncutoff
        Nequiv = self.Nequiv

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
        normal_x, normal_y = get_normals(theta)
        # get C2E (check solution to equivalent density) operator for each level
        E2C_SVDs = []
        E2Cs = []
        for ind in range(tree.levels):
            equiv_to_check = Kernel_Form(small_xs[ind], small_ys[ind], large_xs[ind], large_ys[ind])
            # Add_Pressure_Fix(equiv_to_check, normal_x, normal_y, normal_x, normal_y)
            E2C_SVDs.append(SVD_Solver(equiv_to_check))
            E2Cs.append(equiv_to_check)
        C2E_SVDs = []
        C2Es = []
        for ind in range(tree.levels):
            check_to_equiv = Kernel_Form(large_xs[ind], large_ys[ind], small_xs[ind], small_ys[ind])
            # Add_Pressure_Fix(check_to_equiv, normal_x, normal_y, normal_x, normal_y)
            C2E_SVDs.append(SVD_Solver(check_to_equiv))
            C2Es.append(check_to_equiv)
        # get Collected Equivalent Coordinates for each level
        M2MC = []
        for ind in range(tree.levels-1):
            Kern1 = Kernel_Form(small_xs[ind+1] - 0.5*widths[ind+1], small_ys[ind+1] - 0.5*widths[ind+1], large_xs[ind], large_ys[ind])
            Kern2 = Kernel_Form(small_xs[ind+1] - 0.5*widths[ind+1], small_ys[ind+1] + 0.5*widths[ind+1], large_xs[ind], large_ys[ind])
            Kern3 = Kernel_Form(small_xs[ind+1] + 0.5*widths[ind+1], small_ys[ind+1] - 0.5*widths[ind+1], large_xs[ind], large_ys[ind])
            Kern4 = Kernel_Form(small_xs[ind+1] + 0.5*widths[ind+1], small_ys[ind+1] + 0.5*widths[ind+1], large_xs[ind], large_ys[ind])
            Kern = np.column_stack([Kern1, Kern2, Kern3, Kern4])
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
                        M2Lhere[indx,indy] = Kernel_Form(small_xhere, small_yhere, small_xs[ind], small_ys[ind])
            M2LS.append(M2Lhere)

        precomputations = {
            'M2MC'      : M2MC,
            'L2LC'      : L2LC,
            'M2LS'      : M2LS,
            'small_xs'  : small_xs,
            'small_ys'  : small_ys,
            'large_xs'  : large_xs,
            'large_ys'  : large_ys,
            'E2C_SVDs'  : E2C_SVDs,
            'E2Cs'      : E2Cs,
            'C2E_SVDs'  : C2E_SVDs,
            'C2Es'      : C2Es,
        }

        self.precomputations = precomputations
    def general_precomputations(self, svd_tol=1e-12):
        # build SVD compression
        M2LS = self.precomputations['M2LS']
        M2LFS = [None,]
        for ind in range(1, self.tree.levels):
            M2LF = np.empty([7,7], dtype=object)
            M2L = M2LS[ind]
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        M2LF[i,j] = M2L_Evaluator(M2L[i,j], svd_tol)
            M2LFS.append(M2LF)
        self.precomputations['M2LFS'] = M2LFS
    def build_expansions(self, taux, tauy):
        tree = self.tree
        precomputations = self.precomputations
        M2MC = precomputations['M2MC']
        L2LC = precomputations['L2LC']
        M2LS = precomputations['M2LS']
        M2LFS = precomputations['M2LFS']

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
            # allocate space for the partial multipoles
            anum = Level.n_allocate_density
            partial_multipole = np.zeros([anum, 2*Nequiv], dtype=float)
            resh_partial_multipole = partial_multipole.reshape(anum, 2, Nequiv)
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = M2MC[ind].dot(self.reshaped_multipoles[ind+1].T).T.copy()
                distribute(partial_multipole, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            upwards_pass(tree.x, tree.y, Level.this_density_ind, Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid, taux_ordered, tauy_ordered, resh_partial_multipole, precomputations['large_xs'][ind], precomputations['large_ys'][ind])
            self.multipoles[ind] = partial_multipole_to_multipole(partial_multipole, precomputations, ind)
            resh = (int(anum/4), Nequiv*8)
            self.reshaped_multipoles[ind] = np.reshape(self.multipoles[ind], resh)
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
            Parent_Level = tree.Levels[ind-1]
            if ind == 2:
                self.Partial_Local_Expansions[ind] = np.zeros([16, 2*Nequiv], dtype=float)
            M2Ls = np.zeros([self.multipoles[ind].shape[0], 2*Nequiv])
            largest_cutoff = 0
            for i in range(7):
                for j in range(7):
                    if hasattr(M2LFS[ind][i,j], 'cutoff'):
                        largest_cutoff = max(largest_cutoff, M2LFS[ind][i,j].cutoff)
            # build the interaction lists
            dilists = np.empty([7, 7, Level.n_node], dtype=int)
            build_interaction_list(Level.parent_ind, Parent_Level.colleagues, Parent_Level.children_ind, Level.xmid, Level.ymid, Level.width, Level.this_density_ind, dilists)
            # add up interactions
            work = np.empty([largest_cutoff, self.multipoles[ind].shape[0]], dtype=float)
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        workmat = work[:M2LFS[ind][i,j].cutoff] if hasattr(M2LFS[ind][i,j], 'cutoff') else None
                        M2LFS[ind][i,j](self.multipoles[ind].T, out=M2Ls.T, work=workmat)
                        add_interactions_prepared(M2Ls, self.Partial_Local_Expansions[ind], dilists[i,j], Nequiv)
            # convert partial local expansions to local local_expansions
            self.Local_Expansions[ind] = partial_local_to_local(self.Partial_Local_Expansions[ind], precomputations, ind)
            self.resh_Local_Expansions[ind] = self.Local_Expansions[ind].reshape(self.Local_Expansions[ind].shape[0], 2, Nequiv)
            # move local expansions downwards
            if ind < tree.levels-1:
                doit = Level.compute_downwards
                descendant_level = tree.Levels[ind+1]
                local_expansions = self.Local_Expansions[ind][doit]
                partial_local_expansions = L2LC[ind].dot(local_expansions.T).T
                sorter = np.argsort(Level.children_ind[doit])
                self.Partial_Local_Expansions[ind+1] = partial_local_expansions[sorter].reshape([descendant_level.n_node, 2*Nequiv])
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
        local_expansion_evaluation(x, y, inds, locs, tree.xmids, tree.ymids, Local_Expansions, potu, potv, potp, precomputations['large_xs'], precomputations['large_ys'])
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.taux_ordered, self.tauy_ordered, potu, potv, potp, check_self)
        return potu, potv, potp



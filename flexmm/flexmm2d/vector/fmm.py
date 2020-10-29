import numpy as np
import scipy as sp
import scipy.linalg
import numba
import time
from .tree import Tree
import sys

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
            for k in range(Nequiv):
                PLEs[i,k] += M2Ls[di,k]

def fake_print(*args, **kwargs):
    pass
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

def get_functions(functions):

    source_to_partial_multipole = functions['source_to_partial_multipole']
    local_expansion_to_target = functions['local_expansion_to_target']
    kernel_apply_single = functions['kernel_apply_single']
    kernel_apply_single_check = functions['kernel_apply_single_check']

    @numba.njit(parallel=True, fastmath=True)
    def upwards_pass(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, e1, e2):
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
                    y[bi:ti]-ymid[i], tau[bi:ti], pM[li[i]], e1, e2)

    @numba.njit(parallel=True, fastmath=True)
    def local_expansion_evaluation(tx, ty, inds, locs, xmids, ymids, LEs, pot, e1, e2):
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
            pot[i] = local_expansion_to_target(LEs[ind][loc], x, y, e1[ind], e2[ind])

    @numba.njit(parallel=True, fastmath=True)
    def neighbor_evaluation(tx, ty, sx, sy, inds, locs, binds, tinds, colls, tauo, pot, check):
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
                            pot[i] += kernel_apply_single_check(sx[bind:tind], sy[bind:tind], x, y, tauo[bind:tind])
                        else:
                            pot[i] += kernel_apply_single(sx[bind:tind], sy[bind:tind], x, y, tauo[bind:tind])

    new_functions = {
        'upwards_pass'               : upwards_pass,
        'local_expansion_evaluation' : local_expansion_evaluation,
        'neighbor_evaluation'        : neighbor_evaluation,
    }
    functions.update(new_functions)

    return functions

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

class M2L_Evaluator(object):
    def __init__(self, m2l, svd_tol):
        self.m2l = m2l
        self.svd_tol = svd_tol
        S = np.linalg.svd(self.m2l)
        cuts = np.where(S[1] > svd_tol)[0]
        if len(cuts) > 0:
            cutoff = np.where(S[1] > svd_tol)[0][-1] + 1
            # if this is the case, svd compression probably saves time
            if 2*cutoff < m2l.shape[0]:
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

class FMM(object):
    def __init__(self, x, y, functions, Nequiv=48, Ncutoff=50, iscomplex=False, bbox=None, verbose=False):
        self.x = x
        self.y = y
        self.bbox = bbox
        self.Nequiv = Nequiv
        self.Ncutoff = Ncutoff
        self.functions = functions
        self.dtype = np.complex128 if iscomplex else np.float64
        self.verbose = verbose
        self.print = get_print_function(self.verbose)
        self.build_tree()
    def build_tree(self):
        st = time.time()
        self.tree = Tree(self.x, self.y, self.Ncutoff, self.bbox)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
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
    def build_expansions(self, tau):
        functions = self.functions
        tree = self.tree
        precomputations = self.precomputations
        M2MC = precomputations['M2MC']
        L2LC = precomputations['L2LC']
        M2LS = precomputations['M2LS']
        M2LFS = precomputations['M2LFS']

        partial_multipole_to_multipole = functions['partial_multipole_to_multipole']
        partial_local_to_local = functions['partial_local_to_local']
        upwards_pass = functions['wrapped_upwards_pass']
        Nequiv, Ncutoff = self.Nequiv, self.Ncutoff
        
        tau_ordered = tau[tree.ordv]
        self.tau = tau
        self.tau_ordered = tau_ordered
        self.multipoles = [None,]*tree.levels
        self.reshaped_multipoles = [None,]*tree.levels
        # upwards pass - start at bottom leaf nodes and build multipoles up
        st = time.time()
        for ind in reversed(range(tree.levels)[1:]):
            Level = tree.Levels[ind]
            # allocate space for the partial multipoles
            anum = Level.n_allocate_density
            partial_multipole = np.zeros([anum, Nequiv], dtype=self.dtype)
            self.multipoles[ind] = np.empty([anum, Nequiv], dtype=self.dtype)
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = M2MC[ind].dot(self.reshaped_multipoles[ind+1].T).T
                distribute(partial_multipole, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            upwards_pass(tree.x, tree.y, Level.this_density_ind, Level.compute_upwards, Level.bot_ind, Level.top_ind, Level.xmid, Level.ymid, tau_ordered, partial_multipole, precomputations, ind)
            self.multipoles[ind] = partial_multipole_to_multipole(partial_multipole, precomputations, ind)
            resh = (int(anum/4), int(Nequiv*4))
            self.reshaped_multipoles[ind] = np.reshape(self.multipoles[ind], resh)
            if self.reshaped_multipoles[ind].flags.owndata:
                raise Exception('Something went wrong with reshaping the equivalent densities, it made a copy instead of a view.')
        et = time.time()
        self.print('....Time for upwards pass:      {:0.2f}'.format(1000*(et-st)))
        # downwards pass - start at top and work down to build up local expansions
        st = time.time()
        self.Partial_Local_Expansions = [None,]*tree.levels
        self.Local_Expansions = [None,]*tree.levels
        self.Local_Expansions[0] = np.zeros([1, Nequiv], dtype=self.dtype)
        self.Local_Expansions[1] = np.zeros([4, Nequiv], dtype=self.dtype)
        for ind in range(2, tree.levels):
            Level = tree.Levels[ind]
            Parent_Level = tree.Levels[ind-1]
            if ind == 2:
                self.Partial_Local_Expansions[ind] = np.zeros([16, Nequiv], dtype=self.dtype)
            M2Ls = np.zeros([self.multipoles[ind].shape[0], Nequiv])
            largest_cutoff = 0
            for i in range(7):
                for j in range(7):
                    if hasattr(M2LFS[ind][i,j], 'cutoff'):
                        largest_cutoff = max(largest_cutoff, M2LFS[ind][i,j].cutoff)
            # build the interaction lists
            dilists = np.empty([7, 7, Level.n_node], dtype=int)
            build_interaction_list(Level.parent_ind, Parent_Level.colleagues, Parent_Level.children_ind, Level.xmid, Level.ymid, Level.width, Level.this_density_ind, dilists)
            # add up interactions
            work = np.empty([largest_cutoff, self.multipoles[ind].shape[0]], dtype=tau.dtype)
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        workmat = work[:M2LFS[ind][i,j].cutoff] if hasattr(M2LFS[ind][i,j], 'cutoff') else None
                        M2LFS[ind][i,j](self.multipoles[ind].T, out=M2Ls.T, work=workmat)
                        add_interactions_prepared(M2Ls, self.Partial_Local_Expansions[ind], dilists[i,j], Nequiv)
            # convert partial local expansions to local local_expansions
            self.Local_Expansions[ind] = partial_local_to_local(self.Partial_Local_Expansions[ind], precomputations, ind)
            # move local expansions downwards
            if ind < tree.levels-1:
                doit = Level.compute_downwards
                descendant_level = tree.Levels[ind+1]
                local_expansions = self.Local_Expansions[ind][doit]
                partial_local_expansions = L2LC[ind].dot(local_expansions.T).T
                sorter = np.argsort(Level.children_ind[doit])
                self.Partial_Local_Expansions[ind+1] = partial_local_expansions[sorter].reshape([descendant_level.n_node, Nequiv])
        et = time.time()
        self.print('....Time for downwards pass:    {:0.2f}'.format(1000*(et-st)))
    def evaluate_to_points(self, x,  y, check_self=False):
        functions = self.functions
        precomputations = self.precomputations
        tree = self.tree

        local_expansion_evaluation = functions['wrapped_local_expansion_evaluation']
        neighbor_evaluation = functions['neighbor_evaluation']

        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate local expansions
        pot = np.zeros(x.size, dtype=self.dtype)
        Local_Expansions = self.Local_Expansions
        local_expansion_evaluation(x, y, inds, locs, tree.xmids, tree.ymids, Local_Expansions, pot, precomputations)
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.tau_ordered, pot, check_self)
        return pot



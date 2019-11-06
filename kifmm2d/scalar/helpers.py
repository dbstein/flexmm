import numpy as np
import numba
import scipy as sp
import scipy.linalg
from ..float_dict import FloatDict

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

class Helper(object):
    def __init__(self, helper=None):
        if helper is not None:
            # load compiled functions
            self.functions = helper.functions
            self.evaluators = helper.evaluators
            self.upwardors = helper.upwardors
            # load precomputations
            self.precomputations = helper.precomputations
            # load specific things
            self.load_specific(helper)
        else:
            self.precomputations = FloatDict()
            self.functions = {}
            self.evaluators = {}
            self.upwardors = {}
    def initialize(self, fmm):
        # prepare
        self.prepare(fmm)
        # collect the widths
        self.collect_widths()
    def get_bbox(self, px, py, bbox):
        if bbox is None:
            bbox = [np.min(px), np.max(px), np.min(py), np.max(py)]
        # get the smallest width we know about
        widths = self.collect_widths()
        if len(widths) > 0:
            small_width = np.min(self.collect_widths())
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
        return bbox_x + bbox_y
    def load_specific(self, helper):
        pass
    def get_precomputation(self, width):
        return self.precomputations[width]
    def prepare(self, fmm):
        tree = fmm.tree
        self.small_xs = []
        self.small_ys = []
        self.large_xs = []
        self.large_ys = []        
        for Level in tree.Levels:
            width = Level.width
            if not self.has_precomputation(width):
                precomp = Precomputation(width, fmm)
                self.add_precomputation(precomp)
            precomp = self.get_precomputation(width)
            self.small_xs.append(precomp.small_x)
            self.small_ys.append(precomp.small_y)
            self.large_xs.append(precomp.large_x)
            self.large_ys.append(precomp.large_y)
    def collect_widths(self):
        return np.array(list(self.precomputations), dtype=float)
    def has_precomputation(self, width):
        return width in self.precomputations
    def add_precomputation(self, p):
        self.precomputations[p.width] = p
    def build_base_functions(self, Kernel_Eval):
        if 'kernel_eval' not in self.functions:
            self.functions['kernel_eval'] = Kernel_Eval
        Kernel_Eval = self.functions['kernel_eval']
        if 'kernel_app' not in self.functions:
            @numba.njit(fastmath=True)
            def kernel_app(sx, sy, tx, ty, tau, e):
                return (Kernel_Eval(sx, sy, tx, ty)*tau,)
            self.functions['kernel_app'] = kernel_app
        if 'extra_kernel_app' not in self.functions:
            @numba.njit(fastmath=True)
            def extra_kernel_app(sx, sy, tx, ty, tau, e):
                return (Kernel_Eval(sx, sy, tx, ty)*tau[0],)
            self.functions['extra_kernel_app'] = extra_kernel_app
        if 'kernel_form' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def _kernel_form(sx, sy, tx, ty, out):
                for i in numba.prange(sx.size):
                    for j in range(tx.size):
                        out[j,i] = Kernel_Eval(sx[i], sy[i], tx[j], ty[j])
            def kernel_form(sx, sy, tx=None, ty=None, out=None, mdtype=float):
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
                _kernel_form(sx, sy, tx, ty, out)
                if isself:
                    np.fill_diagonal(out, 0.0)
                return out
            self.functions['kernel_form'] = kernel_form

    def register_upwardor(self, KAS, name):

        if name not in self.upwardors:

            @numba.njit(parallel=True, fastmath=True)
            def upwardor(x, y, li, cu, bind, tind, xmid, ymid, tau, pM, large_x, large_y, extras):
                n_eval = tau.shape[0]
                for i in numba.prange(bind.size):
                    if cu[i] and li[i] >= 0:
                        bi = bind[i]
                        ti = tind[i]
                        pm = pM[li[i]]
                        for j in range(bi, ti):
                            xj = x[j]-xmid[i]
                            yj = y[j]-ymid[i]
                            ej = extras[:,j]
                            tj = tau[:,j]
                            for k in range(large_x.size):
                                pm[k] += KAS(xj, yj, large_x[k], large_y[k], tj, ej)[0]
            self.upwardors[name] = upwardor

    def register_evaluator(self, SLP_APPLY, ALL_APPLY, name, outputs):

        if name not in self.evaluators:

            @numba.njit(parallel=True, fastmath=True)
            def neighbor_evaluation(tx, ty, sx, sy, inds, locs, binds, tinds, colls, tauo, pot, check, extras):
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
                pot,   *[n_eval, nt]    - potential
                check, bool     - whether to check for source/targ coincidences
                """
                n_eval = pot.shape[0]
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
                                for k in range(bind, tind):
                                    if not check or not (sx[k]==x and sy[k]==y):
                                        out = ALL_APPLY(sx[k], sy[k], x, y, tauo[:,k], extras[:,k])
                                        for ni in range(n_eval):
                                            pot[ni, i] += out[ni]

            @numba.njit(parallel=True, fastmath=True)
            def local_expansion_evaluation(tx, ty, inds, locs, xmids, ymids, LEs, pot, large_xs, large_ys, extras):
                n_eval = pot.shape[0]
                for i in numba.prange(tx.size):
                    x = tx[i]
                    y = ty[i]
                    ind = inds[i]
                    loc = locs[i]
                    x = x - xmids[ind][loc]
                    y = y - ymids[ind][loc]
                    LE = LEs[ind][loc]
                    sx = large_xs[ind]
                    sy = large_ys[ind]
                    for k in range(sx.size):
                        out = SLP_APPLY(sx[k], sy[k], x, y, LE[k], extras)
                        for ni in range(n_eval):
                            pot[ni, i] += out[ni]
            
            evaluator = [neighbor_evaluation, local_expansion_evaluation, outputs]
            self.evaluators[name] = evaluator

class Precomputation(object):
    def __init__(self, width, fmm, svd_tol=1e-14):
        self.width = width
        self.svd_tol = svd_tol
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
    def compress(self):
        M2L = self.M2L
        M2LF = np.empty([7,7], dtype=object)
        for i in range(7):
            for j in range(7):
                if not (i in [2,3,4] and j in [2,3,4]):
                    M2LF[i,j] = M2L_Evaluator(M2L[i,j], self.svd_tol)
        # get and store the max cutoff
        largest_cutoff = 0
        for i in range(7):
            for j in range(7):
                if hasattr(M2LF[i,j], 'cutoff'):
                    largest_cutoff = max(largest_cutoff, M2LF[i,j].cutoff)
        self.M2LF = M2LF
        self.largest_cutoff = largest_cutoff


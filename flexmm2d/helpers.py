import numpy as np
import numba
from .float_dict import FloatDict

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
            # load precomputations
            self.precomputations = helper.precomputations
            # load specific things
            self.load_specific(helper)
        else:
            self.precomputations = FloatDict()
            self.functions = {}
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
    def prepare(self):
        raise NotImplementedError
    def collect_widths(self):
        return np.array(list(self.precomputations), dtype=float)
    def has_precomputation(self, width):
        return width in self.precomputations
    def add_precomputation(self, p):
        self.precomputations[p.width] = p
    def get_local_expansion_extras(self):
        raise NotImplementedError
    def build_base_functions(self, Kernel_Eval):
        if 'kernel_eval' not in self.functions:
            self.functions['kernel_eval'] = Kernel_Eval
        Kernel_Eval = self.functions['kernel_eval']
        if 'kernel_apply_single' not in self.functions:
            @numba.njit(fastmath=True)
            def kernel_apply_single(sx, sy, tx, ty, tau):
                u = 0.0
                for i in range(sx.size):
                    u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
                return u
            self.functions['kernel_apply_single'] = kernel_apply_single
        if 'kernel_apply_single_check' not in self.functions:
            @numba.njit(fastmath=True)
            def kernel_apply_single_check(sx, sy, tx, ty, tau):
                u = 0.0
                for i in range(sx.size):
                    if not (tx - sx[i] == 0 and ty - sy[i] == 0):
                        u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
                return u
            self.functions['kernel_apply_single_check'] = kernel_apply_single_check
        if 'kernel_apply' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def kernel_apply(sx, sy, tx, ty, tau, out):
                for j in numba.prange(tx.size):
                    outj = 0.0
                    for i in range(sx.size):
                        outj += Kernel_Eval(sx[i], sy[i], tx[j], ty[j])*tau[i]
                    out[j] = outj
            self.functions['kernel_apply'] = kernel_apply
        if 'kernel_apply_self' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def kernel_apply_self(sx, sy, tau, out):
                for j in numba.prange(sx.size):
                    outj = 0.0
                    for i in range(sx.size):
                        if i != j:
                            outj += Kernel_Eval(sx[i], sy[i], sx[j], sy[j])*tau[i]
                    out[j] = outj
            self.functions['kernel_apply_self'] = kernel_apply_self
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
    def register_neighbor_evaluator(self, kernel_apply_single, name):

        if name not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def neighbor_evaluation(tx, ty, sx, sy, inds, locs, binds, tinds, colls, tauo, pot):
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
                                out = kernel_apply_single(sx[bind:tind], sy[bind:tind], x, y, tauo[bind:tind])
                                if True:
                                    pot[0, i] += out
                                else:
                                    for ni in range(n_eval):
                                        pot[ni, i] += out[ni]
            self.functions[name] = neighbor_evaluation

class Precomputation(object):
    def __init__(self, width, svd_tol=1e-14):
        self.width = width
        self.svd_tol = svd_tol
    def precompute(self):
        raise NotImplementedError
    def get_upwards_extras(self):
        raise NotImplementedError
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


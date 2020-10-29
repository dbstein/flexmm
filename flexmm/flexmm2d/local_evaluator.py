import numpy as np
import scipy as sp
import scipy.linalg
import numba
import time
from ..local_tree import LocalTree
import sys

def fake_print(*args, **kwargs):
    pass
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

class LocalHelper(object):
    def __init__(self, helper=None):
        if helper is not None:
            # load compiled functions
            self.functions = helper.functions
            # load specific things
            self.load_specific(helper)
        else:
            self.functions = {}
    def get_bbox(self, px, py, bbox):
        if bbox is None:
            bbox = [np.min(px), np.max(px), np.min(py), np.max(py)]
        return bbox
    def build_base_functions(self, Kernel_Add):
        if 'kernel_add' not in self.functions:
            self.functions['kernel_add'] = Kernel_Add
        Kernel_Add = self.functions['kernel_add']
        if 'kernel_add_single' not in self.functions:
            @numba.njit(fastmath=True)
            def kernel_add_single(sx, sy, tx, ty, tau, out):
                for i in range(sx.size):
                        Kernel_Add(sx[i], sy[i], tx, ty, tau[i], out)
            self.functions['kernel_add_single'] = kernel_add_single
            @numba.njit(parallel=True, fastmath=True)
            def kernel_add_single_check(sx, sy, tx, ty, tau, out):
                for i in range(sx.size):
                    if not (tx - sx[i] == 0 and ty - sy[i] == 0):
                        Kernel_Add(sx[i], sy[i], tx, ty, tau[i], out)
            self.functions['kernel_add_single_check'] = kernel_add_single_check
        if 'kernel_apply_self' not in self.functions:
            @numba.njit(parallel=True, fastmath=True)
            def kernel_apply_self(sx, sy, tau, out):
                out[:] = 0.0
                for j in numba.prange(sx.size):
                    for i in range(sx.size):
                        if i != j:
                            Kernel_Add(sx[i], sy[i], sx[j], sy[j], tau[i], out[j])
            self.functions['kernel_apply_self'] = kernel_apply_self
    def register_neighbor_evaluator(self, kernel_add_single, name):
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
                                kernel_add_single(sx[bind:tind], sy[bind:tind], x, y, tauo[bind:tind], pot[i])
            self.functions[name] = neighbor_evaluation

class LocalEvaluator(object):
    def __init__(self, x, y, kernel_eval, min_distance, ncutoff=20, dtype=float, bbox=None, helper=LocalHelper(), verbose=False):
        # store inputs
        self.x = x
        self.y = y
        self.kernel_eval = kernel_eval
        self.min_distance = min_distance
        self.ncutoff = ncutoff
        self.dtype = dtype
        self.bbox = bbox
        self.helper = helper
        self.verbose = verbose
        # get print function
        self.print = get_print_function(self.verbose)
        # reset bbox to be compatible with helper
        self.bbox = self.helper.get_bbox(self.x, self.y, self.bbox)
        # build the tree
        self.build_tree()
        # build basic functions
        self.helper.build_base_functions(kernel_eval)
        # register some useful neighbor evaluators
        self.register_neighbor_evaluator(self.helper.functions['kernel_add_single'], 'neighbor_potential_target_evaluation')
        self.register_neighbor_evaluator(self.helper.functions['kernel_add_single_check'], 'neighbor_potential_source_evaluation')
    def build_tree(self):
        st = time.time()
        self.tree = LocalTree(self.x, self.y, self.min_distance, self.ncutoff, self.bbox)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
    def register_neighbor_evaluator(self, kernel_apply_single, name):
        self.helper.register_neighbor_evaluator(kernel_apply_single, name)
    def load_tau(self, tau):
        self.tau = tau
        self.tau_ordered = tau[self.tree.ordv]
    def source_evaluation(self, x, y, out):
        return self.evaluate_to_points(x, y, 'neighbor_potential_source_evaluation', out)
    def target_evaluation(self, x, y, out):
        return self.evaluate_to_points(x, y, 'neighbor_potential_target_evaluation', out)
    def evaluate_to_points(self, x, y, name, out):
        # since we're using only add functions, make sure out is 0...
        out[:] = 0.0
        # access the tree and appropriate evaluator
        tree = self.tree
        neighbor_evaluation = self.helper.functions[name]
        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.tau_ordered, out)


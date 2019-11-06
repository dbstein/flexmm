import numpy as np
import scipy as sp
import scipy.linalg
import numba
import time
from .tree import Tree
from .helpers import Helper
import sys

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
            for k in range(Nequiv):
                PLEs[i,k] += M2Ls[di,k]

def fake_print(*args, **kwargs):
    pass
def myprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def get_print_function(verbose):
    return myprint if verbose else fake_print

class FMM(object):
    def __init__(self, x, y, kernel_eval, Ncutoff, Nequiv, dtype=float, bbox=None, helper=Helper(), verbose=False):
        # store inputs
        self.x = x
        self.y = y
        self.kernel_eval = kernel_eval
        self.Nequiv = Nequiv
        self.Ncutoff = Ncutoff
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
        self.register_evaluator(self.helper.functions['kernel_app'], self.helper.functions['extra_kernel_app'], 'potential_evaluation', 1)
        # register some useful upwardor functions
        self.register_upwardor(self.helper.functions['extra_kernel_app'], 'slp')
        # build some functions
        self.helper.prepare(self)
    def build_tree(self):
        st = time.time()
        self.tree = Tree(self.x, self.y, self.Ncutoff, self.bbox)
        tree_formation_time = (time.time() - st)*1000
        self.print('....Tree formed in:             {:0.1f}'.format(tree_formation_time))
    def upwards_pass(self, tau, name, extras):
        # extract tree, helper, functions
        tree = self.tree
        helper = self.helper
        functions = helper.functions
        Nequiv = self.Nequiv

        if extras is None:
            extras = np.row_stack([self.x, self.y])
        self.extras = extras

        # order and stores tau
        if len(tau.shape) == 1:
            tau = tau.reshape([1, tau.size])
        tau_ordered = tau[:, tree.ordv]
        self.tau = tau
        self.tau_ordered = tau_ordered
        self.extras_ordered = self.extras[:, tree.ordv]
        
        # initialize multipoles
        self.multipoles = [None,]*tree.levels
        self.reshaped_multipoles = [None,]*tree.levels

        # upwards pass - start at bottom leaf nodes and build multipoles up
        st = time.time()
        for ind in reversed(range(tree.levels)[2:]):
            Level = tree.Levels[ind]
            # extract precomputations from helper
            prec = helper.get_precomputation(Level.width)
            # allocate space for the partial multipoles
            anum = Level.n_allocate_density
            partial_multipole = np.zeros([anum, Nequiv], dtype=self.dtype)
            # check if there is a level below us, if there is, lift all its expansions
            if ind != tree.levels-1:
                ancestor_level = tree.Levels[ind+1]
                temp1 = prec.M2MC.dot(self.reshaped_multipoles[ind+1].T).T
                distribute(partial_multipole, temp1, ancestor_level.short_parent_ind, ancestor_level.parent_density_ind, Level.this_density_ind)
            # execute source --> partial_multipole routine
            self.helper.upwardors[name](tree.x, tree.y,
                Level.this_density_ind, Level.compute_upwards, Level.bot_ind,
                Level.top_ind, Level.xmid, Level.ymid, tau_ordered,
                partial_multipole, prec.large_x, prec.large_y, self.extras_ordered)
            # transform this to an actual multipole
            self.multipoles[ind] = sp.linalg.lu_solve(prec.S2L_LU, partial_multipole.T, overwrite_b=True, check_finite=False).T
            # reshape the multipole
            resh = (int(anum/4), int(Nequiv*4))
            self.reshaped_multipoles[ind] = np.reshape(self.multipoles[ind], resh)
            if self.reshaped_multipoles[ind].flags.owndata:
                raise Exception('Reshaped multipoles needs to use multipoles data...')
        et = time.time()
        self.print('....Time for upwards pass:      {:0.2f}'.format(1000*(et-st)))
    def downwards_pass(self):
        # extract tree, helper, functions
        tree = self.tree
        helper = self.helper
        functions = helper.functions
        Nequiv = self.Nequiv

        # downwards pass - start at top and work down to build up local expansions
        st = time.time()
        # initialize local expansions
        self.Partial_Local_Expansions = [None,]*tree.levels
        self.Local_Expansions = [None,]*tree.levels
        self.Local_Expansions[0] = np.zeros([1, Nequiv], dtype=self.dtype)
        self.Local_Expansions[1] = np.zeros([4, Nequiv], dtype=self.dtype)
        self.Partial_Local_Expansions[2] = np.zeros([16, Nequiv], dtype=self.dtype)
        for ind in range(2, tree.levels):
            Level = tree.Levels[ind]
            prec = helper.get_precomputation(Level.width)
            Parent_Level = tree.Levels[ind-1]
            M2Ls = np.zeros([self.multipoles[ind].shape[0], Nequiv])
            # extract the interaction list
            dilists = tree.interaction_lists[ind]
            # add up interactions
            work = np.empty([prec.largest_cutoff, self.multipoles[ind].shape[0]], dtype=self.dtype)
            for i in range(7):
                for j in range(7):
                    if not (i in [2,3,4] and j in [2,3,4]):
                        workmat = work[:prec.M2LF[i,j].cutoff] if hasattr(prec.M2LF[i,j], 'cutoff') else None
                        prec.M2LF[i,j](self.multipoles[ind].T, out=M2Ls.T, work=workmat)
                        add_interactions_prepared(M2Ls, self.Partial_Local_Expansions[ind], dilists[i,j], Nequiv)
            # convert partial local expansions to local local_expansions
            self.Local_Expansions[ind] = sp.linalg.lu_solve(prec.L2S_LU, self.Partial_Local_Expansions[ind].T, overwrite_b=True, check_finite=False).T
            # move local expansions downwards
            if ind < tree.levels-1:
                doit = Level.compute_downwards
                descendant_level = tree.Levels[ind+1]
                local_expansions = self.Local_Expansions[ind][doit]
                partial_local_expansions = prec.L2LC.dot(local_expansions.T).T
                sorter = np.argsort(Level.children_ind[doit])
                self.Partial_Local_Expansions[ind+1] = partial_local_expansions[sorter].reshape([descendant_level.n_node, Nequiv])
        et = time.time()
        self.print('....Time for downwards pass:    {:0.2f}'.format(1000*(et-st)))

    def build_expansions(self, tau, name='slp', extras=None):
        self.upwards_pass(tau, name, extras)
        self.downwards_pass()

    def register_evaluator(self, KAS1, KAS2, name, outputs):
        self.helper.register_evaluator(KAS1, KAS2, name, outputs)
    def register_upwardor(self, KAS, name):
        self.helper.register_upwardor(KAS, name)

    def source_evaluation(self, x, y):
        return self.evaluate_to_points(x, y, 'potential_evaluation', True)
    def target_evaluation(self, x, y):
        return self.evaluate_to_points(x, y, 'potential_evaluation', False)

    def evaluate_to_points(self, x, y, name, check):
        tree = self.tree
        neighbor_evaluation = self.helper.evaluators[name]

        # get level ind, level loc for the point (x, y)
        inds, locs = tree.locate_points(x, y)

        evaluator = self.helper.evaluators[name]
        neighbor_evaluation = evaluator[0]
        local_evaluation = evaluator[1]
        outputs = evaluator[2]
        # evaluate local expansions
        pot = np.zeros([outputs, x.size], dtype=self.dtype)
        local_evaluation(x, y, inds, locs, tree.xmids, tree.ymids,
                self.Local_Expansions, pot, self.helper.large_xs, self.helper.large_ys, self.extras_ordered)
        # evaluate interactions from neighbor cells to (x, y)
        neighbor_evaluation(x, y, tree.x, tree.y, inds, locs, tree.bot_inds, tree.top_inds, tree.colleagues, self.tau_ordered, pot, check, self.extras_ordered)
        return pot


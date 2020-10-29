import numpy as np
import scipy as sp
import numba
import scipy.spatial

"""
Same as tree3, but getting rid of the separate arrays for
the 'fake leaves', hopefully this cleans up the code a bit
"""
cacheit = True
tree_search_crossover = 100

@numba.njit(cache=cacheit)
def classify(x, y, midx, midy, n, cl, ns):
    """
    Determine which 'class' each point belongs to in the current node
    class = 1 ==> lower left  (x in [xmin, xmid], y in [ymin, ymid])
    class = 2 ==> upper left  (x in [xmin, xmid], y in [ymid, ymax])
    class = 3 ==> lower right (x in [xmid, xmax], y in [ymin, ymid])
    class = 4 ==> upper right (x in [xmid, xmax], y in [ymid, ymax])
    inputs: 
        x,    f8[n], x coordinates for node
        y,    f8[n], y coordinates for node
        midx, f8,    x midpoint of node
        midy, f8,    y midpoint of node
        n,    i8,    number of points in node
    outputs:
        cl,   i1[n], class (described above)
        ns,   i8[4], number of points in each class
    """
    for i in range(n):
        highx = x[i] > midx
        highy = y[i] > midy
        cla = 2*highx + highy
        cl[i] = cla
        ns[cla] += 1
@numba.njit(cache=cacheit)
def get_target(i, nns):
    """
    Used in the reordering routine, determines which 'class' we
    actually want in the given index [i]. See more details in the 
    description of reorder_inplace
    inputs:
        i,   i8,    index
        nns, i8[4], cumulative sum of number of points in each class
    returns:
        
    """
    if i < nns[1]:
        target = 0
    elif i < nns[2]:
        target = 1
    elif i < nns[3]:
        target = 2
    else:
        target = 3
    return target
@numba.njit(cache=cacheit)
def swap(x, i, j):
    """
    Inputs:
        x, f8[:]/i1[:]/i8[:], array on which to swap inds
        i, i8,                first index to swap
        j, i8,                second index to swap
    This function just modifies x to swap x[i] and x[j]
    """
    a = x[i]
    x[i] = x[j]
    x[j] = a
@numba.njit(cache=cacheit)
def get_inds(ns):
    """
    Compute cumulative sum of the number of points in each subnode
    Inputs:
        ns,   i8[4], number of points in each subnode
    Outputs:
        inds, i8[4], cumulative sum of ns, ignoring the last part of the sum
                        i.e. if ns=[1,2,3,4], inds==>[0,1,3,6]
    """
    inds = np.empty(4, dtype=np.int64)
    inds[0] = 0
    for i in range(3):
        inds[i+1] = inds[i] + ns[i]
    return inds
@numba.njit(cache=cacheit)
def reorder_inplace(x, y, ordv, midx, midy):
    """
    This function plays an integral role in tree formation and does
        several things:
        1) Determine which points belong to which subnodes
        2) Reorder x, y, and ordv variables
        3) Compute the number of points in each new subnode
    Inputs:
        x,    f8[:], x coordinates in node
        y,    f8[:], y coordinates in node
        ordv, i8[:], ordering variable
        midx, f8,    midpoint (in x) of node being split
        midy, f8,    midpoint (in y) of node being split
    Outputs:
        ns,   i8[4], number of points in each node
    """
    n = x.shape[0]
    cl = np.empty(n, dtype=np.int8)
    ns = np.zeros(4, dtype=np.int64)
    classify(x, y, midx, midy, n, cl, ns)
    inds = get_inds(ns)
    nns = inds.copy()
    for i in range(n):
        target = get_target(i, nns)
        keep_going = True
        while keep_going:
            cli = cl[i]
            icli = inds[cli]
            if cli != target:
                swap(cl, i, icli)
                swap(x, i, icli)
                swap(y, i, icli)
                swap(ordv, i, icli)
                inds[cli] += 1
                if cl[icli] == target:
                    keep_going = False
            else:
                inds[target] += 1
                keep_going = False
    return nns

@numba.njit(parallel=True, cache=cacheit)
def divide_and_reorder(x, y, ordv, tosplit, half_width, xmid, ymid, bot_ind, \
                top_ind, leaf, new_xmin, new_ymin, new_bot_ind,
                new_top_ind, parent_ind, children_ind, children_start_ind, Xlist):
    """
    For every node in a level, check if node has too many points
    If it does, split that node, reordering x, y, ordv variables as we go
    Keep track of information relating to new child nodes formed
    Inputs:
        x,           f8[:], x coordinates (for whole tree)
        y,           f8[:], y coordinates (for whole tree)
        ordv,        i8[:], ordering variable (for whole tree)
        tosplit,     b1[:], whether the given node needs to be split
        half_width,  f8,    half width of current nodes
        xmid,        f8[:], x midpoints of the current nodes
        ymid,        f8[:], y midpoints of the current nodes
        bot_ind,     i8[:], bottom indeces into x/y arrays for current nodes
        top_ind,     i8[:], top indeces into x/y arrays for current nodes
    Outputs:
        leaf,         b1[:], indicator for whether current nodes are leaves
        new_xmin,     f8[:], minimum x values for child nodes
        new_ymin,     f8[:], minimum y values for child nodes
        new_bot_ind,  i8[:], bottom indeces into x/y arrays for current nodes
        new_top_ind,  i8[:], top indeces into x/y arrays for current nodes
        parent_ind,   i8[:], indeces into prior level array for parents
        children_ind, i8[:], indeces into next level array for children
        children_start_ind, i8[:], base value for children_ind, used for additions
        Xlist,        b1: whether this division is for Xlist or not
    """
    num_nodes = xmid.shape[0]
    split_ids = np.zeros(num_nodes, dtype=np.int64)
    split_tracker = 0
    for i in range(num_nodes):
        if tosplit[i]:
            split_ids[i] = split_tracker
            split_tracker += 1
    for i in numba.prange(num_nodes):
        if tosplit[i]:
            split_tracker = split_ids[i]
            bi = bot_ind[i]
            ti = top_ind[i]
            nns = reorder_inplace(x[bi:ti], y[bi:ti], ordv[bi:ti], xmid[i], ymid[i])
            new_xmin[4*split_tracker + 0] = xmid[i] - half_width
            new_xmin[4*split_tracker + 1] = xmid[i] - half_width
            new_xmin[4*split_tracker + 2] = xmid[i]
            new_xmin[4*split_tracker + 3] = xmid[i]
            new_ymin[4*split_tracker + 0] = ymid[i] - half_width
            new_ymin[4*split_tracker + 1] = ymid[i]
            new_ymin[4*split_tracker + 2] = ymid[i] - half_width
            new_ymin[4*split_tracker + 3] = ymid[i]
            new_bot_ind[4*split_tracker + 0] = bot_ind[i] + nns[0]
            new_bot_ind[4*split_tracker + 1] = bot_ind[i] + nns[1]
            new_bot_ind[4*split_tracker + 2] = bot_ind[i] + nns[2]
            new_bot_ind[4*split_tracker + 3] = bot_ind[i] + nns[3]
            new_top_ind[4*split_tracker + 0] = bot_ind[i] + nns[1]
            new_top_ind[4*split_tracker + 1] = bot_ind[i] + nns[2]
            new_top_ind[4*split_tracker + 2] = bot_ind[i] + nns[3]
            new_top_ind[4*split_tracker + 3] = top_ind[i]
            if not Xlist:
                leaf[i] = False
            for j in range(4):
                parent_ind[4*split_tracker + j] = i
            children_ind[i] = children_start_ind + 4*split_tracker

def get_new_level(level, x, y, ordv, mw, ncutoff):
    """
    Split any nodes in level that have more than ppl points
    Into new nodes, reordering x/y/ordv along the way
    And construct new level from each node
    Inputs:
        level, Level
        x,       f8[:], x coordinates (for whole tree)
        y,       f8[:], y coordinates (for whole tree)
        ordv,    i8[:], ordering variable (for whole tree)
        mw,      f8,    minimum width of panels
        ncutoff, i8,    don't refine large panels if they have less than this
    """
    # figure out how many need to be split
    to_split = np.logical_and(level.ns > ncutoff, level.width > 2*mw)
    num_to_split = to_split.sum()
    num_new = 4*num_to_split
    # allocate memory for outputs of divide_and_reorder
    xmin = np.empty(num_new, dtype=float)
    ymin = np.empty(num_new, dtype=float)
    bot_ind = np.empty(num_new, dtype=int)
    top_ind = np.empty(num_new, dtype=int)
    parent_ind = np.empty(num_new, dtype=int)
    # divde current nodes and reorder the x, y, and ordv arrays
    divide_and_reorder(x, y, ordv, to_split, level.half_width, level.xmid, level.ymid, \
            level.bot_ind, level.top_ind, level.leaf, xmin, ymin, bot_ind, top_ind, parent_ind, level.children_ind, 0, False)
    # construct new level
    new_level = Level(xmin, ymin, level.half_width, bot_ind, top_ind, parent_ind)
    # determine whether further refinement is needed
    keep_going = np.any(np.logical_and(new_level.ns > ncutoff, new_level.width > 2*mw))
    return new_level, keep_going

@numba.njit(parallel=True, cache=cacheit)
def numba_tag_colleagues(xmid, ymid, colleagues, dist):
    n = xmid.shape[0]
    dist2 = dist*dist
    for i in numba.prange(n):
        itrack = 0
        for j in range(n):
            dx = xmid[i]-xmid[j]
            dy = ymid[i]-ymid[j]
            d2 = dx*dx + dy*dy
            if d2 < dist2:
                colleagues[i,itrack] = j
                itrack += 1

@numba.njit(parallel=True, cache=cacheit)
def numba_loop_colleagues(xmid, ymid, dist, parent_ind, ancestor_colleagues, 
                                ancestor_leaf, ancestor_child_inds, colleagues):
    n = xmid.shape[0]
    dist2 = dist*dist
    for i in numba.prange(n):
        itrack = 0
        pi = parent_ind[i]
        for j in range(9):
            pij = ancestor_colleagues[pi,j]
            if pij >= 0:
                if not ancestor_leaf[pij]:
                    ck = ancestor_child_inds[pij]
                    for k in range(4):
                        ckk = ck + k                  
                        dx = xmid[i]-xmid[ckk]
                        dy = ymid[i]-ymid[ckk]
                        d2 = dx*dx + dy*dy
                        if d2 < dist2:
                            colleagues[i,itrack] = ckk
                            itrack += 1

def split_bad_leaves(Level, Descendant_Level, x, y, ordv, bads, Xlist):
    num_to_split = bads.sum()
    num_new = 4*num_to_split
    # allocate memory for outputs of divide_and_reorder
    xmin = np.empty(num_new, dtype=float)
    ymin = np.empty(num_new, dtype=float)
    bot_ind = np.empty(num_new, dtype=int)
    top_ind = np.empty(num_new, dtype=int)
    parent_ind = np.empty(num_new, dtype=int)
    # divde current nodes and reorder the x, y, and ordv arrays
    divide_and_reorder(x, y, ordv, bads, Level.half_width, Level.xmid, Level.ymid, \
            Level.bot_ind, Level.top_ind, Level.leaf, xmin, ymin, bot_ind, top_ind, parent_ind, Level.children_ind, Descendant_Level.n_node, Xlist)
    # add these new nodes to the descendent level
    Descendant_Level.add_nodes(xmin, ymin, bot_ind, top_ind, parent_ind, Xlist)
    # retag colleagues of the descendant level
    Descendant_Level.tag_colleagues(Level)

class Level(object):
    """
    Set of nodes all at the same level (with same width...)
    For use in constructing Tree objects
    """
    def __init__(self, xmin, ymin, width, bot_ind, top_ind, parent_ind):
        """
        Inputs:
            xmin,       f8[:], minimum x values for each node
            ymin,       f8[:], minimum y values for each node
            width,      f8,    width of each node (must be same in x/y directions)
            bot_ind,    i8[:], bottom indeces into x/y arrays for current nodes
            top_ind,    i8[:], top indeces into x/y arrays for current nodes
            parent_ind, i8[:], index to find parent in prior level array
        """
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.half_width = 0.5*self.width
        self.bot_ind = bot_ind
        self.top_ind = top_ind
        self.parent_ind = parent_ind
        self.leaf = np.ones(self.xmin.shape[0], dtype=bool)
        self.fake_leaf = np.zeros(self.xmin.shape[0], dtype=bool)
        self.children_ind = -np.ones(self.xmin.shape[0], dtype=int)
        self.basic_computations()
    def basic_computations(self):
        self.ns = self.top_ind - self.bot_ind
        self.xmid = self.xmin + self.half_width
        self.ymid = self.ymin + self.half_width
        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.width
        self.n_node = self.xmin.shape[0]
        self.short_parent_ind = self.parent_ind[::4]
    def tag_colleagues(self, ancestor=None):
        self.colleagues = -np.ones([self.n_node, 9], dtype=int)
        # self.direct_tag_colleagues()
        #### NEED TO FIX THE ANCESTOR TAG to deal with fake leaves...
        if self.n_node < tree_search_crossover:
            self.direct_tag_colleagues()
        elif ancestor is None:
            self.ckdtree_tag_colleagues()
        else:
            self.ancestor_tag_colleagues(ancestor)
    def direct_tag_colleagues(self):
        dist = 1.5*self.width
        numba_tag_colleagues(self.xmid, self.ymid, self.colleagues, dist)
    def ckdtree_tag_colleagues(self):
        dist = 1.5*self.width
        self.construct_midpoint_tree()
        colleague_list = self.midpoint_tree.query_ball_tree(self.midpoint_tree, dist)
        for ind in range(self.n_node):
            clist = colleague_list[ind]
            self.colleagues[ind,:len(clist)] = clist
    def ancestor_tag_colleagues(self, ancestor):
        dist = 1.5*self.width
        leaf_here = ancestor.children_ind < 0
        # numba_loop_colleagues(self.xmid, self.ymid, dist, self.parent_ind, 
            # ancestor.colleagues, ancestor.leaf, ancestor.children_ind, self.colleagues)
        numba_loop_colleagues(self.xmid, self.ymid, dist, self.parent_ind, 
            ancestor.colleagues, leaf_here, ancestor.children_ind, self.colleagues)
    def construct_midpoint_tree(self):
        if not hasattr(self, 'midpoint_tree'):
            midpoint_data = np.column_stack([self.xmid, self.ymid])
            self.midpoint_tree = sp.spatial.cKDTree(midpoint_data)
    def get_depths(self, descendant):
        self.depths = np.zeros(self.n_node, dtype=int)
        if descendant is not None:
            numba_get_depths(self.depths, self.leaf, self.children_ind, descendant.depths)
    def add_nodes(self, xmin, ymin, bot_ind, top_ind, parent_ind, Xlist):
        self.xmin = np.concatenate([self.xmin, xmin])
        self.ymin = np.concatenate([self.ymin, ymin])
        self.bot_ind = np.concatenate([self.bot_ind, bot_ind])
        self.top_ind = np.concatenate([self.top_ind, top_ind])
        self.parent_ind = np.concatenate([self.parent_ind, parent_ind])
        new_leaf_indicator = np.zeros(xmin.shape[0], dtype=bool) if Xlist else np.ones(xmin.shape[0], dtype=bool)
        self.leaf = np.concatenate([self.leaf, new_leaf_indicator])
        new_fake_leaf_indicator = np.ones(xmin.shape[0], dtype=bool) if Xlist else np.zeros(xmin.shape[0], dtype=bool)
        self.fake_leaf = np.concatenate([self.fake_leaf, new_fake_leaf_indicator])
        self.children_ind = np.concatenate([self.children_ind, -np.ones(xmin.shape[0], dtype=int)])
        self.basic_computations()
    def get_Xlist(self):
        self.Xlist = np.zeros(self.n_node, dtype=bool)
        numba_get_Xlist(self.depths, self.colleagues, self.leaf, self.Xlist)
    def add_null_Xlist(self):
        self.Xlist = np.zeros(self.n_node, dtype=bool)

@numba.njit(parallel=True, cache=cacheit)
def numba_get_depths(depths, leaves, children_ind, descendant_depths):
    n = depths.shape[0]
    for i in numba.prange(n):
        if not leaves[i]:
            child_depths = descendant_depths[children_ind[i]:children_ind[i]+4]
            max_child_depth = np.max(child_depths)
            depths[i] = max_child_depth + 1

@numba.njit(parallel=True, cache=cacheit)
def numba_get_bads(depths, colleagues, leaf, bads):
    n = depths.shape[0]
    for i in numba.prange(n):
        if leaf[i]:
            badi = False
            for j in range(9):
                cj = colleagues[i,j]
                if cj >= 0:
                    level_dist = depths[i]-depths[cj]
                    if level_dist > 1 or level_dist < -1:
                        badi = True
            bads[i] = badi

@numba.njit(parallel=True, cache=cacheit)
def numba_get_Xlist(depths, colleagues, leaf, Xlist):
    n = depths.shape[0]
    for i in numba.prange(n):
        if leaf[i]:
            XlistI = False
            for j in range(9):
                cj = colleagues[i,j]
                if cj >= 0 and cj < n:
                    level_dist = depths[i]-depths[cj]
                    if level_dist < 0:
                        XlistI = True
            Xlist[i] = XlistI

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

try:
    from numba.typed import List
    def list_to_typed_list(L):
        TL = List()
        for x in L: TL.append(x)
        return TL
except:
    def list_to_typed_list(L):
        return L

class LocalTree(object):
    """
    Quadtree object for use in computing FMMs
    """
    def __init__(self, x, y, mw, ncutoff, bbox=None):
        """
        Inputs:
            x,       f8[:], x coordinates for which tree will be constructed
            y,       f8[:], y coordinates for which tree will be constructed
            mw,      f8,    minimum width of panels
            ncutoff, i8, stop refining panels if they have less than this
            bbox,    f8[4], [xmin, xmax, ymin, ymax] for eval bbox
        """
        self.x = x.copy()
        self.y = y.copy()
        self.minimum_width = mw
        self.ncutoff = ncutoff
        self.bbox = bbox
        # xmin = self.x.min() if self.bbox is None else min(self.x.min(), bbox[0])
        # xmax = self.x.max() if self.bbox is None else max(self.x.max(), bbox[1])
        # ymin = self.y.min() if self.bbox is None else min(self.y.min(), bbox[2])
        # ymax = self.y.max() if self.bbox is None else max(self.y.max(), bbox[3])
        # mmin = int(np.floor(np.min([xmin, ymin])))
        # mmax = int(np.ceil (np.max([xmax, ymax])))
        # self.xmin = mmin
        # self.xmax = mmax
        # self.ymin = mmin
        # self.ymax = mmax
        self.xmin = bbox[0]
        self.xmax = bbox[1]
        self.ymin = bbox[2]
        self.ymax = bbox[3]
        self.N = self.x.shape[0]
        # vector to allow reordering of density tau
        self.ordv = np.arange(self.N)
        self.Levels = []
        # setup the first level
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        width = self.xmax-self.xmin
        bot_ind_arr = np.array((0,))
        top_ind_arr = np.array((self.N,))
        parent_ind_arr = np.array((-1,))
        level_0 = Level(xminarr, yminarr, width, bot_ind_arr, top_ind_arr, parent_ind_arr)
        self.Levels.append(level_0)
        if self.N > ncutoff and level_0.width > 2*self.minimum_width:
            current_level = level_0
            keep_going = True
            while keep_going:
                new_level, keep_going = get_new_level(current_level, \
                                self.x, self.y, self.ordv, self.minimum_width, self.ncutoff)
                self.Levels.append(new_level)
                current_level = new_level
        self.levels = len(self.Levels)
        # tag colleagues
        self.tag_colleagues()
        # gather depths
        self.gather_depths()
        # perform level restriction
        self.level_restrict()
        # tag and split the Xlist
        self.split_Xlist()
        # get post-processed information
        self.post_process()
        # get aggregated information
        self.leafs         = list_to_typed_list([Level.leaf for Level in self.Levels])
        self.xmids         = list_to_typed_list([Level.xmid for Level in self.Levels])
        self.ymids         = list_to_typed_list([Level.ymid for Level in self.Levels])
        self.bot_inds      = list_to_typed_list([Level.bot_ind for Level in self.Levels])
        self.top_inds      = list_to_typed_list([Level.top_ind for Level in self.Levels])
        self.colleagues    = list_to_typed_list([Level.colleagues for Level in self.Levels])
        self.children_inds = list_to_typed_list([Level.children_ind for Level in self.Levels])
        # build the interaction list
        self.build_interaction_lists()
    def tag_colleagues(self):
        """
        Tag colleagues (neighbors at same level) for every node in tree
        """
        for ind, Level in enumerate(self.Levels):
            ancestor = None if ind == 0 else self.Levels[ind-1]
            Level.tag_colleagues(ancestor)
    def retag_colleagues(self, lev):
        ancestor = None if lev == 0 else self.Levels[lev-1]
        self.Levels[lev].tag_colleagues(ancestor)
    def level_restrict(self):
        new_nodes = 1
        while(new_nodes > 0):
            new_nodes = 0
            for ind in range(1, self.levels-2):
                Level = self.Levels[ind]
                Descendant_Level = self.Levels[ind+1]
                bads = np.zeros(Level.n_node, dtype=bool)
                numba_get_bads(Level.depths, Level.colleagues, Level.leaf, bads)
                num_bads = np.sum(bads)
                split_bad_leaves(Level, Descendant_Level, self.x, self.y, self.ordv, bads, False)
                self.gather_depths()
                new_nodes += num_bads
    def split_Xlist(self):
        for ind in range(self.levels):
            Level = self.Levels[ind]
            if ind == 0 or ind == self.levels-1:
                Level.add_null_Xlist()
            else:
                Level.get_Xlist()
                Xlist = Level.Xlist
                num_Xlist = np.sum(Xlist)
                if num_Xlist > 0:
                    Descendant_Level = self.Levels[ind+1]
                    split_bad_leaves(Level, Descendant_Level, self.x, self.y, self.ordv, Xlist, True)
    def post_process(self):
        for Level in self.Levels:
            Level.not_leaf = np.logical_not(Level.leaf)
            Level.compute_upwards = np.logical_or(np.logical_and(Level.leaf, np.logical_not(Level.Xlist)), Level.fake_leaf)
            ns = Level.top_ind - Level.bot_ind
            # if its a leaf (or fake leaf) and there are sources, we need to compute a Density
            Level.compute_upwards = np.logical_and(Level.compute_upwards, ns > 0)
            # whether to do things on the downward pass (nicer with Level.not_leaf, reorganize sometime)
            Level.compute_downwards = np.logical_and(np.logical_or(Level.not_leaf, Level.Xlist), np.logical_not(Level.fake_leaf))
            # if there's any points in the node (whether a leaf or not, there should be a non-null Density)
            Level.has_source = ns > 0
            if Level is not self.Levels[0]:
                # whether any siblings have any
                Level.sib_has_source = np.add.reduceat(Level.has_source, np.arange(0, len(Level.has_source), 4)) > 0
                Level.n_sib_has_source = np.sum(Level.sib_has_source)
                # whether to allocate memory for Equiv Densities
                Level.allocate_density = np.repeat(Level.sib_has_source, 4)
                Level.n_allocate_density = np.sum(Level.allocate_density)
                # get the parent ind corresponding to the short density
                Level.parent_density_ind = np.zeros(int(Level.n_node/4), dtype=int) - 1
                Level.parent_density_ind[Level.sib_has_source] = np.arange(Level.n_sib_has_source).astype(int)
                # get this level ind corresponding to check sources
                Level.this_density_ind = np.zeros(int(Level.n_node), dtype=int) - 1
                Level.this_density_ind[Level.allocate_density] = np.arange(Level.n_allocate_density).astype(int)
    def gather_depths(self):
        for ind, Level in reversed(list(enumerate(self.Levels))):
            descendant = None if ind==self.levels-1 else self.Levels[ind+1]
            Level.get_depths(descendant)
    def build_interaction_lists(self):
        self.interaction_lists = [None,]*self.levels
        for ind in range(2, self.levels):
            Level = self.Levels[ind]
            Parent_Level = self.Levels[ind-1]
            dilists = np.empty([7, 7, Level.n_node], dtype=int)
            build_interaction_list(Level.parent_ind, Parent_Level.colleagues, Parent_Level.children_ind, Level.xmid, Level.ymid, Level.width, Level.this_density_ind, dilists)
            self.interaction_lists[ind] = dilists

    """
    Information functions
    """
    def plot(self, ax, mpl, points=False, **kwargs):
        """
        Create a simple plot to visualize the tree
        Inputs:
            ax,     axis, required: on which to plot things
            mpl,    handle to matplotlib import
            points, bool, optional: whether to also scatter the points
        """
        if points:
            ax.scatter(self.x, self.y, color='red', **kwargs)
        lines = []
        clines = []
        for level in self.Levels:
            nleaves = np.sum(level.leaf)
            xls = level.xmin[level.leaf]
            xhs = level.xmax[level.leaf]
            yls = level.ymin[level.leaf]
            yhs = level.ymax[level.leaf]
            lines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
        lc = mpl.collections.LineCollection(lines, colors='black')
        ax.add_collection(lc)
        try:
            for ind in range(1, self.levels-1):
                level = self.Levels[ind]
                nxlist = np.sum(level.Xlist)
                xls = level.xmin[level.Xlist]
                xms = level.xmid[level.Xlist]
                xhs = level.xmax[level.Xlist]
                yls = level.ymin[level.Xlist]
                yms = level.ymid[level.Xlist]
                yhs = level.ymax[level.Xlist]
                clines.extend([[(xms[i], yls[i]), (xms[i], yhs[i])] for i in range(nxlist)])
                clines.extend([[(xls[i], yms[i]), (xhs[i], yms[i])] for i in range(nxlist)])
            clc = mpl.collections.LineCollection(clines, colors='gray', alpha=0.25)
            ax.add_collection(clc)
        except:
            pass
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
    def plot_level(self, ax, mpl, level, **kwargs):
        """
        Create a simple plot to visualize a level of a tree
        Inputs:
            ax,     axis, required: on which to plot things
            mpl,    handle to matplotlib import
        """
        if not isinstance(level, int) or level < 1:
            raise Exception('Level must be an integer and at least 1')
        lines = []
        clines = []
        lev = self.Levels[level-1]
        n_node = lev.n_node
        xls = lev.xmin
        xhs = lev.xmax
        yls = lev.ymin
        yhs = lev.ymax
        # plot the edges
        lines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(n_node)])
        lines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(n_node)])
        lines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(n_node)])
        lines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(n_node)])
        lc = mpl.collections.LineCollection(lines, color='black', linewidth=3)
        ax.add_collection(lc)
        # now add colleagues...
        lines = []
        for i in range(n_node):
            this_x_mid = lev.xmid[i]
            this_y_mid = lev.ymid[i]
            col = lev.colleagues[i]
            col = col[col != -1]
            col = col[col != i]
            ll = [[(this_x_mid, this_y_mid), (lev.xmid[j], lev.ymid[j])] for j in col]
            lines.extend(ll)
        lc = mpl.collections.LineCollection(lines, linewidth=1)
        ax.add_collection(lc)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
    def print_structure(self):
        """
        Prints the stucture of the array (# levels, # leaves per level)
        """
        for ind, Level in enumerate(self.Levels):
            print('Level', ind+1, 'of', self.levels, 'has', Level.n_node, 'nodes.')
    def check_bounding_box(self, x, y):
        ok1 = np.all(x < self.xmin)
        ok2 = np.all(x > self.xmax)
        ok3 = np.all(y < self.ymin)
        ok4 = np.all(y > self.ymax)
        return ok1 and ok2 and ok3 and ok4
    def locate_point(self, x, y):
        """
        Given a point x, y in the bounding box of the tree
        Find 
        """
        self.check_bounding_box(x, y)
        ind = 0
        loc = 0
        Level = self.Levels[ind]
        while not Level.leaf[loc]:
            child_loc = Level.children_ind[loc]
            if x > Level.xmid[loc]: child_loc += 2
            if y > Level.ymid[loc]: child_loc += 1
            loc = child_loc
            ind += 1
            Level = self.Levels[ind]
        return ind, loc
    def locate_points(self, x, y):
        x = x.ravel()
        y = y.ravel()
        self.check_bounding_box(x, y)
        inds = np.zeros(x.size, dtype=int)
        locs = np.zeros(x.size, dtype=int)
        numba_locate_points(x, y, self.leafs, self.xmids, self.ymids, self.children_inds, inds, locs)
        return inds.reshape(x.shape), locs.reshape(x.shape)

@numba.njit(parallel=True, fastmath=True)
def numba_locate_points(tx, ty, leafs, xmids, ymids, cinds, inds, locs):
    n = tx.size
    for i in numba.prange(n):
        x = tx[i]
        y = ty[i]
        ind = 0
        loc = 0
        while not leafs[ind][loc]:
            cloc = cinds[ind][loc]
            if x > xmids[ind][loc]: cloc += 2
            if y > ymids[ind][loc]: cloc += 1
            loc = cloc
            ind += 1
        inds[i] = ind
        locs[i] = loc








import numpy as np
from flexmm.flexmm2d.local_evaluator_category import LocalEvaluator, LocalHelper

def shift_to_interval(v, interval):
    """
    periodically shift values into interval
    """
    r = interval[1] - interval[0]
    return (v - interval[0]) % r + interval[0]

def append(x, y, c, bounds, min_distance):
    xran = bounds[1] - bounds[0]
    yran = bounds[3] - bounds[2]
    # reflect those things over x...
    close_l = x < bounds[0] + min_distance
    close_r = x > bounds[1] - min_distance
    close_b = y < bounds[2] + min_distance
    close_t = y > bounds[3] - min_distance
    middle_x = ~np.logical_or(close_l, close_r)
    middle_y = ~np.logical_or(close_b, close_t)
    close_lb = np.logical_and(close_l, close_b)
    close_lt = np.logical_and(close_l, close_t)
    close_lo = np.logical_and(close_l, middle_y)
    close_rb = np.logical_and(close_r, close_b)
    close_rt = np.logical_and(close_r, close_t)
    close_ro = np.logical_and(close_r, middle_y)
    close_bo = np.logical_and(close_b, middle_x)
    close_to = np.logical_and(close_t, middle_x)
    i = np.arange(x.size)
    new_x = [x,]
    new_y = [y,]
    new_c = [c,]
    new_i = [i,]
    # close on left only
    new_x.append(x[close_lo]+xran)
    new_y.append(y[close_lo])
    new_c.append(c[close_lo])
    new_i.append(i[close_lo])
    # close on right only
    new_x.append(x[close_ro]-xran)
    new_y.append(y[close_ro])
    new_c.append(c[close_ro])
    new_i.append(i[close_ro])
    # close on bottom only
    new_x.append(x[close_bo])
    new_y.append(y[close_bo]+yran)
    new_c.append(c[close_bo])
    new_i.append(i[close_bo])
    # close on top only
    new_x.append(x[close_to])
    new_y.append(y[close_to]-yran)
    new_c.append(c[close_to])
    new_i.append(i[close_to])
    # close on left and bottom
    new_x.append(x[close_lb]+xran)
    new_y.append(y[close_lb])
    new_c.append(c[close_lb])
    new_i.append(i[close_lb])
    new_x.append(x[close_lb]+xran)
    new_y.append(y[close_lb]+yran)
    new_c.append(c[close_lb])
    new_i.append(i[close_lb])
    new_x.append(x[close_lb])
    new_y.append(y[close_lb]+yran)
    new_c.append(c[close_lb])
    new_i.append(i[close_lb])
    # close on left and top
    new_x.append(x[close_lt]+xran)
    new_y.append(y[close_lt])
    new_c.append(c[close_lt])
    new_i.append(i[close_lt])
    new_x.append(x[close_lt]+xran)
    new_y.append(y[close_lt]-yran)
    new_c.append(c[close_lt])
    new_i.append(i[close_lt])
    new_x.append(x[close_lt])
    new_y.append(y[close_lt]-yran)
    new_c.append(c[close_lt])
    new_i.append(i[close_lt])
    # close on right and top
    new_x.append(x[close_rt]-xran)
    new_y.append(y[close_rt])
    new_c.append(c[close_rt])
    new_i.append(i[close_rt])
    new_x.append(x[close_rt]-xran)
    new_y.append(y[close_rt]-yran)
    new_c.append(c[close_rt])
    new_i.append(i[close_rt])
    new_x.append(x[close_rt])
    new_y.append(y[close_rt]-yran)
    new_c.append(c[close_rt])
    new_i.append(i[close_rt])
    # close on right and bottom
    new_x.append(x[close_rb]-xran)
    new_y.append(y[close_rb])
    new_c.append(c[close_rb])
    new_i.append(i[close_rb])
    new_x.append(x[close_rb]-xran)
    new_y.append(y[close_rb]+yran)
    new_c.append(c[close_rb])
    new_i.append(i[close_rb])
    new_x.append(x[close_rb])
    new_y.append(y[close_rb]+yran)
    new_c.append(c[close_rb])
    new_i.append(i[close_rb])
    # push these all back together
    xout = np.concatenate(new_x)
    yout = np.concatenate(new_y)
    cout = np.concatenate(new_c)
    iout = np.concatenate(new_i)
    # return
    return xout, yout, cout, iout

class PeriodicLocalEvaluator(object):
    def __init__(self, x, y, cat, kernel_eval, min_distance, bounds, ncutoff=20, dtype=float, helper=None, verbose=False):
        """
        Periodic local evaluator --- assumes local evaluation distance is less than box size...
        """
        # store inputs
        self.x = shift_to_interval(x, bounds[0:2])
        self.y = shift_to_interval(y, bounds[2:4])
        self.cat = cat
        self.kernel_eval = kernel_eval
        self.min_distance = min_distance
        self.bounds = bounds
        self.ncutoff = ncutoff
        self.dtype = dtype
        self.helper = LocalHelper() if helper is None else helper
        self.verbose = verbose
        # reset bbox to be compatible with helper
        xmin = self.bounds[0]-self.min_distance
        xmax = self.bounds[1]+self.min_distance
        ymin = self.bounds[2]-self.min_distance
        ymax = self.bounds[3]+self.min_distance
        self.bbox = [xmin, xmax, ymin, ymax]
        # append to x, y, cat...
        self.append_x, self.append_y, self.append_cat, self.append_ind = append(self.x, self.y, self.cat, self.bounds, self.min_distance)
        self.LocalEvaluator = LocalEvaluator(self.append_x, self.append_y, self.append_cat, self.kernel_eval, self.min_distance, self.ncutoff, self.dtype, self.bbox, self.helper, self.verbose)
    def load_tau(self, tau):
        append_tau = tau[self.append_ind]
        self.LocalEvaluator.load_tau(append_tau)
    def target_evaluation(self, x, y, cat, out):
        x = shift_to_interval(x, self.bounds[0:2])
        y = shift_to_interval(y, self.bounds[2:4])
        return self.LocalEvaluator.evaluate_to_points(x, y, cat, 'neighbor_potential_target_evaluation', out)

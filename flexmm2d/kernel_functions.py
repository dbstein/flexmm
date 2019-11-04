import numpy as np
import numba

def get_functions(Kernel_Eval):

    # used by everybody
    @numba.njit(fastmath=True)
    def kernel_apply_single(sx, sy, tx, ty, tau):
        u = 0.0
        for i in range(sx.size):
            u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
        return u

    # used by everybody
    @numba.njit(fastmath=True)
    def kernel_apply_single_check(sx, sy, tx, ty, tau):
        u = 0.0
        for i in range(sx.size):
            if not (tx - sx[i] == 0 and ty - sy[i] == 0):
                u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
        return u

    # used by everybody
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

    # used by KIFMM
    @numba.njit(parallel=True, fastmath=True)
    def kernel_apply(sx, sy, tx, ty, tau, out):
        for j in numba.prange(tx.size):
            outj = 0.0
            for i in range(sx.size):
                outj += Kernel_Eval(sx[i], sy[i], tx[j], ty[j])*tau[i]
            out[j] = outj

    # used only for test purposes
    @numba.njit(parallel=True, fastmath=True)
    def kernel_apply_self(sx, sy, tau, out):
        for j in numba.prange(sx.size):
            out[j] = 0.0
            for i in range(sx.size):
                if i != j:
                    out[j] += Kernel_Eval(sx[i], sy[i], sx[j], sy[j])*tau[i]

    kernel_functions = {
        'kernel_apply_single'       : kernel_apply_single,
        'kernel_apply_single_check' : kernel_apply_single_check,
        'kernel_form'               : kernel_form,
        'kernel_apply'              : kernel_apply,
        'kernel_apply_self'         : kernel_apply_self,
    }

    return kernel_functions

def add_gradient_functions(Kernel_Gradient_Eval, kernel_functions):

    @numba.njit(fastmath=True)
    def kernel_gradient_apply_single(sx, sy, tx, ty, tau):
        u  = 0.0
        ux = 0.0
        uy = 0.0
        for i in range(sx.size):
            out = Kernel_Gradient_Eval(sx[i], sy[i], tx, ty)
            u  += out[0]*tau[i]
            ux += out[1]*tau[i]
            uy += out[2]*tau[i]
        return u, ux, uy

    @numba.njit(fastmath=True)
    def kernel_gradient_apply_single_check(sx, sy, tx, ty, tau):
        u  = 0.0
        ux = 0.0
        uy = 0.0
        for i in range(sx.size):
            if not (tx - sx[i] == 0 and ty - sy[i] == 0):
                out = Kernel_Gradient_Eval(sx[i], sy[i], tx, ty)
                u  += out[0]*tau[i]
                ux += out[1]*tau[i]
                uy += out[2]*tau[i]
        return u, ux, uy

    kernel_gradient_functions = {
        'kernel_gradient_apply_single'       : kernel_gradient_apply_single,
        'kernel_gradient_apply_single_check' : kernel_gradient_apply_single_check,
    }

    kernel_functions.update(kernel_gradient_functions)
    return kernel_functions


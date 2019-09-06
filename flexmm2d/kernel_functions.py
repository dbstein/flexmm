import numpy as np
import numba

def get_functions(Kernel_Eval):

    @numba.njit(parallel=True, fastmath=True)
    def kernel_form(sx, sy, tx, ty, out):
        for i in numba.prange(sx.size):
            for j in range(tx.size):
                out[j,i] = Kernel_Eval(sx[i], sy[i], tx[j], ty[j])

    @numba.njit(parallel=True, fastmath=True)
    def kernel_apply(sx, sy, tx, ty, tau, out):
        out[:] = 0.0
        kernel_add(sx, sy, tx, ty, tau, out)

    @numba.njit(parallel=True, fastmath=True)
    def kernel_add(sx, sy, tx, ty, tau, out):
        for j in numba.prange(tx.size):
            for i in range(sx.size):
                out[j] += Kernel_Eval(sx[i], sy[i], tx[j], ty[j])*tau[i]

    @numba.njit(fastmath=True)
    def kernel_apply_single(sx, sy, tx, ty, tau):
        u = 0.0
        for i in range(sx.size):
            u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
        return u

    @numba.njit(fastmath=True)
    def kernel_apply_single_check(sx, sy, tx, ty, tau):
        u = 0.0
        for i in range(sx.size):
            if not (tx - sx[i] == 0 and ty - sy[i] == 0):
                u += Kernel_Eval(sx[i], sy[i], tx, ty)*tau[i]
        return u

    @numba.njit(parallel=True, fastmath=True)
    def kernel_apply_self(sx, sy, tau, out):
        out[:] = 0.0
        kernel_add_self(sx, sy, tau, out)

    @numba.njit(parallel=True, fastmath=True)
    def kernel_add_self(sx, sy, tau, out):
        for j in numba.prange(sx.size):
            for i in range(sx.size):
                if i != j:
                    out[j] += Kernel_Eval(sx[i], sy[i], sx[j], sy[j])*tau[i]

    kernel_functions = {
        'kernel_form'               : kernel_form,
        'kernel_apply'              : kernel_apply,
        'kernel_add'                : kernel_add,
        'kernel_apply_single'       : kernel_apply_single,
        'kernel_apply_single_check' : kernel_apply_single_check,
        'kernel_apply_self'         : kernel_apply_self,
        'kernel_add_self'           : kernel_add_self,
    }

    return kernel_functions

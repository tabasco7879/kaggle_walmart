import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)

def g_cost_fun5(np.ndarray[DTYPE_t, ndim=2] fmat_weight, \
                np.ndarray[DTYPE_t, ndim=2] fmat, \
                np.ndarray[DTYPE_t, ndim=2] Y_hat, \
                np.ndarray[DTYPE_t, ndim=2] D):
    cdef int n=fmat.shape[0]
    cdef int m=fmat.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] g = np.zeros([n,m], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] working_fmat=fmat_weight * fmat
    cdef np.ndarray[DTYPE_t, ndim=2] Fmat=np.dot(working_fmat, working_fmat.T)
    cdef np.ndarray[DTYPE_t, ndim=1] nm=np.linalg.norm(working_fmat, axis=1) # shape=nx1
    cdef int i, j, k
    cdef DTYPE_t d0, c1, c2
    for i in range(n):
        for k in range(m):
            if fmat[i, k]<1e-10:
                continue
            for j in range(n):
                if i==j:
                    continue
                d0=D[i, j]/(nm[j]*(nm[i]**2))
                c1=fmat[i, k] * working_fmat[j, k] * nm[i]
                c2=Fmat[i, j] * fmat[i, k] * working_fmat[i, k] / nm[i]
                g[i, k]+=(c1-c2)*d0
    return g
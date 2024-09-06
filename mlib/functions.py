import ctypes
from .marray import Marray, CMarray

def random(lo, hi, size):
    Marray._C.random_marray.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    Marray._C.random_marray.restype = ctypes.POINTER(CMarray)
    data = Marray._C.random_marray(ctypes.c_double(lo), ctypes.c_double(hi), ctypes.c_int(size))
    res = Marray(children=[True], req_grad=False)
    res.marray = data
    res.shape = [size]
    res.req_grad = False
    res.ndim = 1
    return res

def assert_close(marr1, marr2, precision=1e-3):
    Marray._C.assert_close.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray), ctypes.c_double]
    Marray._C.assert_close.restype = None
    Marray._C.assert_close(marr1.marray, marr2.marray, ctypes.c_double(precision))


def linespace(lo, hi, num_samples):
    Marray._C.linespace_marray.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    Marray._C.linespace_marray.restype = ctypes.POINTER(CMarray)
    data = Marray._C.linespace_marray(ctypes.c_double(lo), ctypes.c_double(hi), ctypes.c_int(num_samples))
    res = Marray(children=[True], req_grad=False)
    res.marray = data
    res.shape = [num_samples]
    res.req_grad = False
    res.ndim = 1
    return res

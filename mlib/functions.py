import ctypes
from .marray import Marray, CMarray

def arange(hi, shape):
    hi = ctypes.c_int(hi)
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*shape.copy())
    Marray._C.arange_marray.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.arange_marray.restype = ctypes.POINTER(CMarray)
    data = Marray._C.arange_marray(hi, cshape, cndim)
    res = Marray(children=[True], req_grad=False)
    res.marray = data
    res.shape = shape
    res.req_grad = False
    res.ndim = len(shape)
    return res

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

def zeros(*shape):
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*list(shape).copy())
    Marray._C.zeros.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.zeros.restype = ctypes.POINTER(CMarray)
    data = Marray._C.zeros(cshape, cndim)
    res = Marray(children=[True], req_grad=False)
    res.marray = data
    res.shape = shape
    res.req_grad = False
    res.ndim = len(shape)
    return res

def ones(*shape):
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*list(shape).copy())
    Marray._C.ones.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.ones.restype = ctypes.POINTER(CMarray)
    data = Marray._C.ones(cshape, cndim)
    res = Marray(children=[1], req_grad=False)
    res.marray = data
    res.shape = shape
    res.ndim = len(shape)
    return res

def eye(n, ndim):
    cndim = ctypes.c_int(ndim)
    cn = ctypes.c_int(n)
    Marray._C.eye_marray.argtypes = [ctypes.c_int, ctypes.c_int]
    Marray._C.eye_marray.restype = ctypes.POINTER(CMarray)
    data = Marray._C.eye_marray(cn, cndim)
    res = Marray(children=[1], req_grad=False)
    res.marray = data
    res.shape = [n] * ndim
    res.req_grad = False
    res.ndim = ndim
    return res
     
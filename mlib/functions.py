import ctypes
from .marray import Marray, CMarray

def scal_prod(list):
    prod = 1
    for l in list:
        prod *= l
    return prod

def arange(hi, shape):
    hi = ctypes.c_int(hi)
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*shape.copy())
    Marray._C.arange_marray.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.arange_marray.restype = ctypes.POINTER(CMarray)
    data = Marray._C.arange_marray(hi, cshape, cndim)
    res = Marray()
    res.marray = data
    res.shape = shape
    res.ndim = len(shape)
    return res

def assert_close(marr1, marr2, precision=1e-2):
    Marray._C.assert_close.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray), ctypes.c_double]
    Marray._C.assert_close.restype = ctypes.POINTER(CMarray)
    precision = ctypes.c_double(precision)
    Marray._C.assert_close(marr1.marray, marr2.marray, precision)

def zeros(*shape):
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*list(shape).copy())
    Marray._C.zeros.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.zeros.restype = ctypes.POINTER(CMarray)
    data = Marray._C.zeros(cshape, cndim)
    res = Marray(children=[1])
    res.marray = data
    res.shape = shape
    res.ndim = len(shape)
    return res

def ones(*shape):
    cndim = ctypes.c_int(len(shape))
    cshape = (ctypes.c_int * len(shape))(*list(shape).copy())
    Marray._C.ones.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    Marray._C.ones.restype = ctypes.POINTER(CMarray)
    data = Marray._C.ones(cshape, cndim)
    res = Marray(children=[1])
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
    res = Marray(children=[1])
    res.marray = data
    res.shape = [n] * ndim
    res.ndim = ndim
    return res
     
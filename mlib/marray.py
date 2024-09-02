import ctypes
import os
from .autograd import *

class CMarray(ctypes.Structure):
    _fields_ = [
        ('storage', ctypes.POINTER(ctypes.c_double)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]

class Marray:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL("./mlib/libmarray.so")

    def __init__(self, data=None, children=None, req_grad=True):

        self.marray = None
        self.grad = None
        self.grad_fn = None
        self.children = children
        self.shape = None
        self.ndim = None
        self.req_grad = req_grad

        if isinstance(data, (int, float)):
            data = [data]

        if data != None:
            data, shape = self.flatten_list(data)
            self.shape = tuple(shape.copy())
            self.ndim = len(shape)
           
            self.data_ctype = (ctypes.c_double * len(data))(*data.copy())
            self.shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
            self.ndim_ctype = ctypes.c_int(len(shape))
        
            Marray._C.create_marray.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            Marray._C.create_marray.restype = ctypes.POINTER(CMarray)

            self.marray = Marray._C.create_marray(
                self.data_ctype,
                self.shape_ctype,
                self.ndim_ctype,
            )

    def flatten_list(self, data):

        shape = []
        tmp = data
        while isinstance(tmp, list):
            shape.append(len(tmp))
            for i in range(len(tmp) - 1):
                if isinstance(tmp[i], list) and isinstance(tmp[i + 1], list) and len(tmp[i]) != len(tmp[i + 1]):
                    raise Exception("ragged marrays are not allowed")
            tmp = tmp[0]

        def flatten_recur(data, res, sh):
            if isinstance(data, list):
                for d in data:
                    flatten_recur(d, res, sh)
            else: res.append(data)

        flattened = []
        flatten_recur(data, flattened, shape)
        return flattened, shape
        
    def __del__(self):
        if hasattr(self, 'data_ctype') and self.data_ctype is not None:

            # python manages the matrix storage (data) and shape if it doesn't have children
            if self.children and len(self.children) > 0:
                Marray._C.delete_storage.argtypes = [ctypes.POINTER(CMarray)]
                Marray._C.delete_storage.restype = None
                Marray._C.delete_storage(self.marray)

                Marray._C.delete_shape.argtypes = [ctypes.POINTER(CMarray)]
                Marray._C.delete_shape.restype = None
                Marray._C.delete_shape(self.marray)

            Marray._C.delete_strides.argtypes = [ctypes.POINTER(CMarray)]
            Marray._C.delete_strides.restype = None
            Marray._C.delete_strides(self.marray)

        if self.marray:
            Marray._C.delete_marray.argtypes = [ctypes.POINTER(CMarray)]
            Marray._C.delete_marray.restype = None
            Marray._C.delete_marray(self.marray)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            pass
       
        if len(indices) != len(self.shape):
            raise Exception('indices must be same length as shape')
        
        indices_array = (ctypes.c_int * len(indices))(*indices)
        Marray._C.get_item.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(ctypes.c_int)]
        Marray._C.get_item.restype = ctypes.c_double
        return Marray._C.get_item(self.marray, indices_array)
    
    def __str__(self):
        if not self.marray:
            return ""
        
        def recur_helper(indices, length_indices):
            # base case
            if length_indices == self.ndim:

                if indices[-1] == self.shape[-1] - 1: return str(self[indices])
    
                return str(self[indices]) +  ", "
                
            current = "["
            for i in range(self.shape[length_indices]):
                current += recur_helper(indices + [i], length_indices + 1)
                      
            if not indices or indices[-1] == self.shape[length_indices - 1] - 1:
                return current + "]"
            
            return current + "],"
        
        return recur_helper([], 0)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = self.ones_like() * other
        Marray._C.elem_add_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray)]
        Marray._C.elem_add_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.elem_add_marray(self.marray, other.marray)
        res = Marray(children=[self, other])
        res.marray = data
        res.grad_fn = ElemAdd(self, other)
        res.ndim = self.ndim
        res.shape = self.shape
        return res
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + -1 * other
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __div__(self, other):
        return self + () * other
    
    def __rdiv__(self, other):
        return self.__div__(other)
    
    def __mul__(self, other):

        res = Marray(children=[self, other])
        if isinstance(other, (int, float)):
            #handle scalar
            cother = ctypes.c_double(other)
            Marray._C.scale_mul_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.c_double]
            Marray._C.scale_mul_marray.restype = ctypes.POINTER(CMarray)
            data = Marray._C.scale_mul_marray(self.marray, cother)
            res.grad_fn = ScalMul(self, other)
            res.marray = data
        else:
            Marray._C.elem_mul_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray)]
            Marray._C.elem_mul_marray.restype = ctypes.POINTER(CMarray)
            data = Marray._C.elem_mul_marray(self.marray, other.marray)
            res.grad_fn = ElemMul(self, other)
            res.marray = data
        
        res.shape = self.shape
        res.ndim = self.ndim
        return res
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other):
        Marray._C.matmul_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray)]
        Marray._C.matmul_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.matmul_marray(self.marray, other.marray)
        res = Marray(children=[self, other])
        res.grad_fn = MatMul(self, other)
        res.marray = data
        res.shape = [self.shape[0], other.shape[1]]
        res.ndim = self.ndim
        return res
    
    def scal_prod(self, list):
        prod = 1
        for l in list:
            prod *= l
        return prod
        
    def flatten(self):
        Marray._C.flatten_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.flatten_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.flatten_marray(self.marray)
        res = Marray(children=[self])
        res.marray = data
        res.shape = [self.scal_prod(self.shape)]
        res.ndim = 1
        return res
    
    def squeeze(self):
        Marray._C.squeeze_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.squeeze_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.squeeze_marray(self.marray)
        res = Marray(children=[self])
        res.marray = data
        res.shape = [max(self.shape)]
        res.ndim = 1
        return res
    
    def unsqueeze(self):
        Marray._C.unsqueeze_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.unsqueeze_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.unsqueeze_marray(self.marray)
        res = Marray(children=[self])
        res.marray = data
        res.shape = [1] + self.shape
        res.ndim = len(res.shape)
        return res
        
    def __repr__(self):
        return self.__str__()
    
    @property
    def T (self):
        Marray._C.transpose.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.transpose.restype = ctypes.POINTER(CMarray)
        data = Marray._C.transpose(self.marray)
        marray = Marray(children=[self])
        marray.marray = data
        marray.ndim = self.ndim
        marray.shape = self.shape[::-1]
        return marray
    
    def reshape(self, *shape):
        cndim = ctypes.c_int(len(shape))
        cshape = (ctypes.c_int * len(shape))(*list(shape).copy())
        Marray._C.reshape.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Marray._C.reshape.restype = ctypes.POINTER(CMarray)
        data = Marray._C.reshape(self.marray, cshape, cndim)
        res = Marray(children=[self])
        res.marray = data
        res.shape = shape
        res.ndim = len(shape)
        return res
    
    def backward(self, grad):
        if not self.req_grad:
            return
        if not self.grad:
            self.grad = grad
        else:
            self.grad += grad
        if not self.grad_fn:
            return
        grads = self.grad_fn.backward(grad)
        for input, input_grad in zip(self.grad_fn.inputs, grads):
            input.backward(input_grad)

    def zeros_like(self):
        Marray._C.zeros_like.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.zeros_like.restype = ctypes.POINTER(CMarray)
        data = Marray._C.zeros_like(self.marray)
        res = Marray(children=True)
        res.marray = data
        res.shape = self.shape
        res.ndim = self.ndim
        return res
    
    def ones_like(self):
        Marray._C.ones_like.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.ones_like.restype = ctypes.POINTER(CMarray)
        data = Marray._C.ones_like(self.marray)
        res = Marray(children=True)
        res.marray = data
        res.shape = self.shape
        res.ndim = self.ndim
        return res
    
    def inverse(self):
        Marray._C.invert_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.invert_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.invert_marray(self.marray)
        res = Marray(children=[self])
        res.marray = data
        res.shape = self.shape
        res.ndim = self.ndim
        return res



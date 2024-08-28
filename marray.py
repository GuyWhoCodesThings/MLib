import ctypes
import os
import grad

PATH_TO_LIB = "./lib"

class CMarray(ctypes.Structure):
    _fields_ = [
        ('storage', ctypes.POINTER(ctypes.c_float)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int),
    ]


class Marray:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL(PATH_TO_LIB + "/libmarray.so")

    def __init__(self, data=None, children=None, req_grad=True):

        self.marray = None
        self.grad = None
        self.grad_fn = None
        self.children = children
        self.grad_fn_name = "" 
        self.shape = None
        self.ndim = None
        self.req_grad = req_grad

        if isinstance(data, (int, float)):
            data = [data]

        if data != None:
            data, shape = self.flatten_list(data)
            self.shape = shape.copy()
            self.ndim = len(shape)
           
            self.data_ctype = (ctypes.c_float * len(data))(*data.copy())
            self.shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
            self.ndim_ctype = ctypes.c_int(len(shape))
        
            Marray._C.create_marray.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            Marray._C.create_marray.restype = ctypes.POINTER(CMarray)

            self.marray = Marray._C.create_marray(
                self.data_ctype,
                self.shape_ctype,
                self.ndim_ctype,
            )

    def flatten_list(self, data):
        if isinstance(data[0], list):
            flat = []
            for l in data:
                flat += l
            return flat, [len(data), len(data[0])]
        else:
            return data, [len(data)]
        
    def __del__(self):
        if hasattr(self, 'data_ctype') and self.data_ctype is not None:

            # python manages the matrix storage (data) and shape if it doesn't have children
            if self.children:
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
        Marray._C.get_item.restype = ctypes.c_float
        return Marray._C.get_item(self.marray, indices_array)
    
    def __str__(self):
        if not self.marray:
            return ""
        
        res = "Marray(["
        if self.ndim == 1:
            for i in range(self.shape[0]):
                res += str(self[[i]]) + ', ' 
            
        else:
            for i in range(self.shape[0]):
                row = "["
                for j in range(self.shape[1]):
                    row += str(self[[i, j]]) + ', '
                row = row[:-2] + '], '
                res += row
        res =  res[:-2] + f"], grad={self.grad}, op={self.grad_fn_name if self.grad_fn_name else 'None'})"
        return res
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = self.ones_like() * other
        Marray._C.elem_add_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray)]
        Marray._C.elem_add_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.elem_add_marray(self.marray, other.marray)
        marray = Marray(children=True)
        marray.marray = data
        marray.ndim = self.ndim
        marray.shape = self.shape
        return marray
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + -1 * other
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):

        res = Marray(children=True)
        if isinstance(other, (int, float)):
            #handle scalar
            other = ctypes.c_float(other)
            Marray._C.scale_mul_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.c_float]
            Marray._C.scale_mul_marray.restype = ctypes.POINTER(CMarray)
            data = Marray._C.scale_mul_marray(self.marray, other)
            res.marray = data
        else:
            Marray._C.elem_mul_marray.argtypes = [ctypes.POINTER(CMarray), ctypes.POINTER(CMarray)]
            Marray._C.elem_mul_marray.restype = ctypes.POINTER(CMarray)
            data = Marray._C.elem_mul_marray(self.marray, other.marray)
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
        res =Marray(children=True)
        res.marray = data
        res.shape = [self.shape[0], other.shape[1]]
        res.ndim = self.ndim
        return res
        
    def flatten(self):
        Marray._C.flatten_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.flatten_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.flatten_marray(self.marray)
        res = Marray(children=True)
        res.marray = data
        res.shape = [scal_prod(self.shape)]
        res.ndim = 1
        return res
    
    def squeeze(self):
        Marray._C.squeeze_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.squeeze_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.squeeze_marray(self.marray)
        res = Marray(children=True)
        res.marray = data
        res.shape = [max(self.shape)]
        res.ndim = 1
        return res
    
    def unsqueeze(self):
        Marray._C.unsqueeze_marray.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.unsqueeze_marray.restype = ctypes.POINTER(CMarray)
        data = Marray._C.unsqueeze_marray(self.marray)
        res = Marray(children=True)
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
        marray = Marray(children=True)
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
        res = Marray()
        res.marray = data
        res.shape = shape
        res.ndim = len(shape)
        return res

    
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

def zeros_like(marr):
        Marray._C.zeros_like.argtypes = [ctypes.POINTER(CMarray)]
        Marray._C.zeros_like.restype = ctypes.POINTER(CMarray)
        data = Marray._C.zeros_like(marr.marray)
        res = Marray(children=True)
        res.marray = data
        res.shape = marr.shape
        res.ndim = marr.ndim
        return res
    
def ones_like(marr):
    Marray._C.ones_like.argtypes = [ctypes.POINTER(CMarray)]
    Marray._C.ones_like.restype = ctypes.POINTER(CMarray)
    data = Marray._C.ones_like(marr.marray)
    res = Marray(children=True)
    res.marray = data
    res.shape = marr.shape
    res.ndim = marr.ndim
    return res
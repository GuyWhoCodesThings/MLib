import ctypes
import os
import grad

PATH_TO_LIB = "./lib/libmatrix.so"

class CMatrix(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_double)),
        ('rows', ctypes.c_int),
        ('cols', ctypes.c_int),
    ]

class Matrix:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL(PATH_TO_LIB)

    def __init__(self, data=None, req_grad=True):

        self.matrix = None
        self.grad = None
        self.grad_fn = None
        self.children = None
        self.grad_fn_name = ""
        self.req_grad = req_grad

        if isinstance(data, (int, float)):
            data = [data]

        if data != None:
            data, rows, cols = self.flatten(data)
            self._data_ctype = (ctypes.c_double * len(data))(*data)
            self._rows_ctype = ctypes.c_int(rows)
            self._cols_ctype = ctypes.c_int(cols)
        
            Matrix._C.create_matrix.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
            Matrix._C.create_matrix.restype = ctypes.POINTER(CMatrix)

            self.matrix = Matrix._C.create_matrix(
                self._data_ctype,
                self._rows_ctype,
                self._cols_ctype
            )
            
    def flatten(self, data):
        if isinstance(data[0], list):
            flat = []
            for l in data:
                flat += l
            return flat, len(data), len(data[0])
        else:
            return data, 1, len(data)
        
    def __del__(self):
        if hasattr(self, '_data_ctype') and self._data_ctype is not None:
            Matrix._C.delete_data.argtypes = [ctypes.POINTER(CMatrix)]
            Matrix._C.delete_data.restype = None
            Matrix._C.delete_data(self.matrix)
        if self.matrix:
            Matrix._C.delete_matrix.argtypes = [ctypes.POINTER(CMatrix)]
            Matrix._C.delete_matrix.restype = None
            Matrix._C.delete_matrix(self.matrix)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            pass
        if len(indices) == 1:
            row = indices[0]
        if len(indices) > 2:
            raise ValueError('indices must be length 2')
        row, col = indices
        Matrix._C.get_item.argtypes = [ctypes.POINTER(CMatrix), ctypes.c_int, ctypes.c_int]      
        Matrix._C.get_item.restype = ctypes.c_double   
        return Matrix._C.get_item(self.matrix, ctypes.c_int(row), ctypes.c_int(col))
    
    def __str__(self):
        if self.matrix:
            res = "Mat(["
            for i in range(self.matrix.contents.rows):
                row = "["
                for j in range(self.matrix.contents.cols):
                    row += str(self[tuple([i, j])]) + ", "
                row = row[:-2] + "], "
                res += row
            res =  res[:-2] + f"], grad={self.grad}, op={self.grad_fn_name if self.grad_fn_name else 'None'})"
            return res
        return "matrix is None"

    def __repr__(self):
        return self.__str__()
        
    def reshape(self, new_shape):
        pass

    def __add__(self, other):
        if isinstance(other, Matrix):
            Matrix._C.elem_add_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.POINTER(CMatrix)]      
            Matrix._C.elem_add_matrix.restype = ctypes.POINTER(CMatrix)
            res_matrix = Matrix._C.elem_add_matrix(self.matrix, other.matrix)
            res = Matrix()
            res.matrix = res_matrix
            res.grad_fn_name = "ElemAdd"
            res.grad_fn = lambda g: grad.add_backward(res)
            return res
        
        Matrix._C.scal_add_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.c_double]      
        Matrix._C.scal_add_matrix.restype = ctypes.POINTER(CMatrix)
        res_matrix = Matrix._C.scal_add_matrix(self.matrix, ctypes.c_double(other))
        res = Matrix()
        res.matrix = res_matrix
        res.grad_fn_name = "ScalAdd"
        res.grad_fn = lambda g: grad.add_backward(res)
        return res
    def __sub__(self, other):
        if isinstance(other, Matrix):
            Matrix._C.elem_sub_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.POINTER(CMatrix)]      
            Matrix._C.elem_sub_matrix.restype = ctypes.POINTER(CMatrix)
            res_matrix = Matrix._C.elem_sub_matrix(self.matrix, other.matrix)
            res = Matrix()
            res.matrix = res_matrix
            res.grad_fn_name = "ElemAdd"
            res.grad_fn = lambda g: grad.sub_backward(res)
            return res
        
        Matrix._C.scal_sub_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.c_double]      
        Matrix._C.scal_sub_matrix.restype = ctypes.POINTER(CMatrix)
        res_matrix = Matrix._C.scal_sub_matrix(self.matrix, ctypes.c_double(other))
        res = Matrix()
        res.matrix = res_matrix
        res.grad_fn_name = "ScalSub"
        res.grad_fn = lambda g: grad.sub_backward(res)
        return res
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            Matrix._C.elem_mul_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.POINTER(CMatrix)]      
            Matrix._C.elem_mul_matrix.restype = ctypes.POINTER(CMatrix)
            res_matrix = Matrix._C.elem_mul_matrix(self.matrix, other.matrix)
            res = Matrix()
            res.matrix = res_matrix
            res.grad_fn_name = "ElemAdd"
            res.grad_fn = lambda g: grad.elem_mul_backward(res, other)
            return res
        
        Matrix._C.scal_mul_matrix.argtypes = [ctypes.POINTER(CMatrix), ctypes.c_double]      
        Matrix._C.scal_mul_matrix.restype = ctypes.POINTER(CMatrix)
        res_matrix = Matrix._C.scal_mul_matrix(self.matrix, ctypes.c_double(other))
        res = Matrix()
        res.matrix = res_matrix
        res.grad_fn_name = "ScalSub"
        res.grad_fn = lambda g: grad.scal_mul_backward(res, self, other)
        return res
    
    @property
    def T (self):
        Matrix._C.transpose.argtypes = [ctypes.POINTER(CMatrix)]
        Matrix._C.transpose.restype = ctypes.POINTER(CMatrix)
        res_matrix = Matrix._C.transpose(self.matrix)
        res = Matrix()
        res.matrix = res_matrix
        res.grad_fn_name = "Trans"
        res.grad_fn = lambda g: grad.trans_backward(res, self)
        return res


    
    def backward(self, grad=1):
        if not self.req_grad:
            return
        if not self.grad:
            self.grad = grad
        else:
            self.grad += grad
        
        if not self.grad_fn:
            return
        
        grads = self.grad_fn(self.grad)
        for node, grad in zip(self.children, grads):
            node.backward(grad)
    
        

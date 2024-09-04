class GradFn:
    def __init__(self, name, *inputs):
        self.name = name
        self.inputs = inputs

class ElemAdd(GradFn):
    def __init__(self, x, y):
        super().__init__("ElemAdd", x, y)
    def backward(self, grad):
        return [grad, grad]
    
class ElemMul(GradFn):
    def __init__(self, x, y):
        super().__init__("ElemMul", x, y)
    def backward(self, grad):
        x, y = self.inputs
        return [y * grad, x * grad]
    
class ScalMul(GradFn):
    def __init__(self, x, c):
        super().__init__("ScalMul", x)
        self.c = c
    def backward(self, grad):
        return [grad * self.c]

class MatMul(GradFn):
    def __init__(self, x, y):
        super().__init__("MatMul", x,y)
    def backward(self, grad):
        x, y = self.inputs
        return [grad @ y.T, x.T @ grad]

class Trans(GradFn):
    def __init__(self, x):
        super().__init__("Trans", x)
    def backward(self, grad):
        return [grad.T]
    
class Inverse(GradFn):
    def __init__(self, x):
        super().__init__("Inverse", x)
    def backward(self, grad):
        x = self.inputs[0]
        inv_x = x.inverse()  # Assuming a method to compute the inverse of x
        return [(-1 * inv_x.T) @ grad @ inv_x.T]
    
class Reshape(GradFn):
    def __init__(self, x):
        super().__init__("Reshape", x)
        self.old_shape = x.shape
    def backward(self, grad):
        # Reshape the gradient to the shape of the original tensor
        return [grad.reshape(*self.old_shape)]


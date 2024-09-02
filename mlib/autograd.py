class GradFn:
    def __init__(self, *inputs):
        self.name = None
        self.inputs = inputs

class ElemAdd(GradFn):
    def __init__(self, x, y):
        self.name = "ElemAdd"
        super().__init__(x, y)
    def backward(self, grad):
        return [grad, grad]
    
class ElemMul(GradFn):
    def __init__(self, x, y):
        self.name = "ElemMul"
        super().__init__(x, y)
    def backward(self, grad):
        x, y = self.inputs
        return [y * grad, x * grad]
    
class ScalMul(GradFn):
    def __init__(self, x, c):
        self.name = "ScalMul"
        super().__init__(x)
        self.c = c
    def backward(self, grad):
        return [grad * self.c]

class MatMul(GradFn):
    def __init__(self, x, y):
        self.name = "MatMul"
        super().__init__(x,y)
    def backward(self, grad):
        x, y = self.inputs
        return [grad @ y.T, x.T @ grad]
    


class Trans(GradFn):
    def __init__(self, x):
        self.name = "Trans"
        super().__init__(x)
    def backward(self, grad):
        return [grad.T]

# C = A @ B
# dC/dB = A
# dC/dA = B.T
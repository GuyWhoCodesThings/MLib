from marray import *

a = Marray([[1],[3]])
b = Marray([[1,2]])
grad = Marray([[1.0]])
c = b @ a
c.backward(grad)
print(c)
print(c.grad)
print(a)
print(a.grad)
print(b)
print(b.grad)

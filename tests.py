from marray import Marray, CMarray

a = Marray([1,2,3])
print(a)
b = a.unsqueeze()
print(b)
c = b.T
print(c)
d = b @ c
print(d)




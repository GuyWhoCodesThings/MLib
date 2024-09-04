import mlib

a = mlib.Marray([[1,2,3]])
b = a.T
print(a,b)
c = (a @ b)
print(c.shape)
d = c.flatten()
d.backward()

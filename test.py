import mlib
import mlib.functions as F
import time
import random

x = mlib.Marray([[[[0,1,2,4]]]])
print(x)
y = x[0,0,0]
y = y + 1
print(y)
print(x)
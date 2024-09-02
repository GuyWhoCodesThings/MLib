import mlib
import mlib.functions as F
import time
import random
import numpy as np
from numpy.linalg import inv

f = lambda x: 4 * x + random.uniform(0, 2)
data = [0.1, 0.5, -1.2, 2.1]
out = [f(d) for d in data]

x = mlib.Marray([data]).T
y_true = mlib.Marray([out]).T
print(x)
print(y_true)

w = (x.T @ x).inverse() @ x.T @ y_true
print(w)


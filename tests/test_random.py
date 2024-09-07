import unittest
import mlib
import mlib.functions

class TestRandom(unittest.TestCase):

    def test_random(self):
        try:
            x = mlib.random(0,1,1000)
            size = 1
            for s in x.shape:
                size *= s
            for i in range(size):
                assert x[i] >= 0 and x[i] <= 1
        except:
            raise
    
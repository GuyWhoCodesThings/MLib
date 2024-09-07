import unittest
import mlib
import mlib.functions

class TestReshape(unittest.TestCase):

    def test_reshape_sm(self):
        try:
            x = mlib.random(0,10, 10).reshape(2,5)
            y = mlib.Marray([[1,2,3,4,5], [1,2,3,4,5]])
            _ = x + y
        except:
            raise
    
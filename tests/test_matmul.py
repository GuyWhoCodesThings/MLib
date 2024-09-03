import unittest
import mlib
import mlib.functions

class TestTensorOperations(unittest.TestCase):

    def test_matmul_sm(self):
        try:
            x = mlib.Marray([[1,1,1,1]])
            y = mlib.Marray([[1],[1],[1],[1]])
            z = mlib.Marray([[4]])
            mlib.functions.assert_close(x @ y, z)
        except:
            raise
    def test_matmul_sm2(self):
        try:
            x = mlib.Marray([[1,1,1]])
            y = mlib.Marray([[1],[1],[1]])
            z = mlib.Marray([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
            mlib.functions.assert_close(y @ x, z)
        except:
            raise
    
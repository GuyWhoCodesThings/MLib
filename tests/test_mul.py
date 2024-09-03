import unittest
import mlib
import mlib.functions


class TestTensorOperations(unittest.TestCase):

    def test_mul_1d(self):
        try:
            x = mlib.Marray([1,2,3])
            y = mlib.Marray([2,3,4])
            z = mlib.Marray([2,6,12])
            mlib.functions.assert_close(x * y, z)
        except:
            raise
    def test_mul_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = mlib.Marray([[[2,3,4]]])
            z = mlib.Marray([[[2,6,12]]])
            mlib.functions.assert_close(x * y, z)
        except:
            raise
    def test_scal_mul_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = 2
            z = mlib.Marray([[[2,4,6]]])
            mlib.functions.assert_close(x * y, z)
        except:
            raise
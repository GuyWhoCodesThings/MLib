import unittest
import mlib
import mlib.functions


class TestDiv(unittest.TestCase):

    def test_div_1d(self):
        try:
            x = mlib.Marray([2,6,12])
            y = mlib.Marray([1,2,3])
            z = mlib.Marray([2,3,4])
            mlib.functions.assert_close(x / y, z)
        except:
            raise
    def test_div_3d(self):
        try:
            x = mlib.Marray([[[2,6,12]]])
            y = mlib.Marray([[[1,2,3]]])
            z = mlib.Marray([[[2,3,4]]])
            mlib.functions.assert_close(x / y, z)
        except:
            raise
    def test_div_mul_3d(self):
        try:
            x = mlib.Marray([[[2,4,6]]])
            y = 2
            z = mlib.Marray([[[1,2,3]]])
            mlib.functions.assert_close(x / y, z)
        except:
            raise
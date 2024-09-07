import unittest
import mlib
import mlib.functions


class TestSub(unittest.TestCase):

    def test_sub_1d(self):
        try:
            x = mlib.Marray([1,2,3])
            y = mlib.Marray([2,3,4])
            z = mlib.Marray([-1,-1,-1])
            mlib.functions.assert_close(x - y, z)
        except:
            raise
    def test_sub_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = mlib.Marray([[[2,3,4]]])
            z = mlib.Marray([[[-1,-1,-1]]])
            mlib.functions.assert_close(x - y, z)
        except:
            raise
    def test_scal_sub_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = 2
            z = mlib.Marray([[[-1,0,1]]])
            mlib.functions.assert_close(x - y, z)
        except:
            raise
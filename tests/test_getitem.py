import unittest
import mlib
import mlib.functions

class TestGetItem(unittest.TestCase):

    def test_getitem_scalar(self):
        try:
            x = mlib.Marray([1,2,3,4])
            y = 2.0
            assert x[1] == y
        except:
            raise
    def test_getitem_1d(self):
        try:
            x = mlib.Marray([[1,2],[3,4]])
            y = mlib.Marray([3,4])
            mlib.assert_close(x[1], y)
        except:
            raise
    def test_getitem_2d(self):
        try:
            x = mlib.Marray([[[1,2],[3,4]]])
            y = mlib.Marray([[1,2],[3,4]])
            mlib.assert_close(x[0], y)
        except:
            raise
    
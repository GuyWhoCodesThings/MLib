import unittest
import mlib
import mlib.functions


class TestAdd(unittest.TestCase):

    def test_add_1d(self):
        try:
            x = mlib.Marray([1,2,3])
            y = mlib.Marray([2,3,4])
            z = mlib.Marray([3,5,7])
            mlib.functions.assert_close(x + y, z)
        except:
            raise
    def test_add_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = mlib.Marray([[[2,3,4]]])
            z = mlib.Marray([[[3,5,7]]])
            mlib.functions.assert_close(x + y, z)
        except:
            raise
    def test_scal_add_3d(self):
        try:
            x = mlib.Marray([[[1,2,3]]])
            y = 2
            z = mlib.Marray([[[3,4,5]]])
            mlib.functions.assert_close(x + y, z)
        except:
            raise

    def test_grad_add(self):
        try:
            x = mlib.Marray([1])
            y = mlib.Marray([2])
            z = x + y
            z.backward()
            mlib.assert_close(z.grad, x.grad)
            mlib.assert_close(x.grad, y.grad)
        except:
            raise
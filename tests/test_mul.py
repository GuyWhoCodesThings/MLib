import unittest
import mlib
import mlib.functions


class TestMul(unittest.TestCase):

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
    def test_grad_mul(self):
        try:
            x = mlib.Marray([1])
            z = x * 2
            z.backward()
            grad = mlib.Marray([2])
            mlib.functions.assert_close(x.grad, grad)
        except:
            raise
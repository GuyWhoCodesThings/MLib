import unittest
import mlib

class TestInverse(unittest.TestCase):

    def test_inverse_eye(self):
        try:
            x = mlib.eye(3,2)
            x_inv = x.inverse()
            mlib.assert_close(x, x_inv)
        except:
            raise
    
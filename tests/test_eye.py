import unittest
import mlib
import mlib.functions

class TestEye(unittest.TestCase):

    def test_eye_2d(self):
        try:
            x = mlib.eye(3,2)
            y = mlib.Marray([[1,0,0],[0,1,0],[0,0,1]])
            mlib.assert_close(x, y)
  
        except:
            raise
    
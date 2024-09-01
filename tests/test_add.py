from unittest import TestCase
import mlib
import mlib.functions as F
import mlib.functions
class TestAdd(TestCase):
    def test_add_1D(self):
        try:
            a = mlib.Marray([1,2,3,4])
            b = mlib.Marray([-1,-2,-2,-2])
            actual = mlib.Marray([0,0,1,2])
            # mlib.functions.assert_close(a + b, actual)
        except: 
            raise

    # def test_add_2D(self):
    #     try:
    #         a = Marray([[1,2],[3,4]])
    #         b = Marray([[1,2], [-3,4]])
    #         actual = Marray([[2,4], [0,8]])
    #         assert_close(a + b, actual)
    #         print('passed')
    #     except: 
    #         raise

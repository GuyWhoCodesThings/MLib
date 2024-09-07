import unittest
import mlib
import mlib.functions

class TestLinespace(unittest.TestCase):

    def test_linespace_sm(self):
        try:
            x = mlib.linespace(0,1,5)
            y = mlib.Marray([0.0, 0.25, 0.5, 0.75, 1.0])     
            mlib.functions.assert_close(x, y)
        except:
            raise
    
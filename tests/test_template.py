# -*- coding:utf-8 -*-
import unittest


class BaseClassTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # preparations which have to be made only once
        pass

    # all test cases must start with "test..."
    def test_x(self):
        pass

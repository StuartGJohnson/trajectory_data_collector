import unittest
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        thing = [True] * 4
        print(thing)
        print(all(thing))

if __name__ == '__main__':
    unittest.main()

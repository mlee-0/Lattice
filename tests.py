"""Run this file to perform unit tests to ensure functions work as intended."""


import unittest

import numpy as np

from preprocessing import *


class TestReadingDataset(unittest.TestCase):
    def setUp(self):
        self.struts = read_struts()

    def test_read_coordinates(self):
        coordinates = read_coordinates()
        self.assertTrue(all(len(_) == 3 for _ in coordinates), "Each set of coordinates must contain 3 values.")
        self.assertEqual(len(coordinates), np.prod(INPUT_SHAPE), "Incorrect number of coordinate points.")

    def test_mask_of_active_nodes(self):
        nodes = np.arange(27).reshape((3, 3, 3)) + 1
        
        mask = mask_of_active_nodes([2, 4, 5], [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], nodes)
        self.assertEqual(mask.sum(), 6)
        self.assertTrue(np.all(mask == (nodes == 3) + (nodes == 4) + (nodes == 7) + (nodes == 8) + (nodes == 9) + (nodes == 10)))

        mask = mask_of_active_nodes([1, 2, 3, 4], [[10, 8], [10, 9], [10, 11], [10, 12], [11, 9]], nodes)
        self.assertEqual(mask.sum(), 5)


if __name__ == '__main__':
    unittest.main()
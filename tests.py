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

class TestHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def test_make_struts(self):
        length = 10
        size = 11
        struts = make_struts(length=length, shape=(size, size, size))
        directions = []

        for node_1, node_2 in struts:
            dx, dy, dz = [coordinate_2 - coordinate_1 for coordinate_1, coordinate_2 in zip(node_1, node_2)]
            # Check for zero-length directions.
            self.assertFalse(dx == 0 and dy == 0 and dz == 0)
            # Check that coordinates increase from node 1 to node 2.
            self.assertTrue(all([dx >= 0, dy >= 0, dz >= 0]))
            # Check for symmetry along each dimension.
            for coordinate_1, coordinate_2,  in zip(node_1, node_2):
                self.assertLessEqual(abs((coordinate_1 - 0) - (size - coordinate_2)), 1)

            directions.append((dx, dy, dz))
        
        # Check for the correct number of struts.
        self.assertEqual(len(directions), (length+1) ** 3 - 1)
        # Check for duplicates.
        self.assertEqual(len(directions), len(set(directions)))


if __name__ == '__main__':
    unittest.main()
"""Test for DNC memory and addressing
"""
import unittest

import torch

from scratch.layers.dnc.memory import Memory


class TestDNCMemory(unittest.TestCase):
    """Test for DNC memory and addressing
    """

    def setUp(self):
        self.read_heads = 2
        self.memory = Memory(self.read_heads)

    def test_get_write_weight(self):
        pass

    def test_get_content_addressing(self):
        memory = torch.DoubleTensor([
            1, -1, -1, 1, 4, 5, 6, 7, -1, 1, 1, -1, 12, 16, 14, 10, 1, -1, 1,
            -1, -1, 1, -1, 1
        ]).view(2, 3, 4)
        keys = torch.DoubleTensor(
            [4, 5, 6, 7, 4, 5, 6, 7, 12, 16, 14, 10, 12, 16, 14, 12]).view(
                2, self.read_heads, 4)
        strength = torch.DoubleTensor([1, 2, 2, 1]).view(2, self.read_heads, 1)
        addressing = self.memory._get_content_addressing(keys, strength, memory)
        self.assertAlmostEqual(torch.sum(addressing[0, 0]), 1)
        self.assertAlmostEqual(torch.sum(addressing[0, 1]), 1)
        self.assertAlmostEqual(torch.sum(addressing[1, 0]), 1)
        self.assertAlmostEqual(torch.sum(addressing[1, 1]), 1)
        self.assertTrue(addressing[0, 0, 1] > addressing[0, 0, 0])
        self.assertTrue(addressing[0, 0, 1] > addressing[0, 0, 2])
        self.assertTrue(addressing[1, 0, 0] > addressing[1, 0, 1])
        self.assertTrue(addressing[1, 0, 0] > addressing[1, 0, 2])
        self.assertTrue(addressing[1, 0, 0] > addressing[1, 1, 0])

    def test_update_read_weight(self):
        pass

    def test_get_allocation_weight(self):
        usage = torch.arange(1, 6, dtype=torch.double).view(1, 1, 5)
        usage /= 6
        order = torch.randperm(5)
        usage = usage[:, :, order]
        _, rev_idx = torch.sort(order)
        allocation = self.memory._get_allocation_weight(usage)
        allocation = allocation[:, :, rev_idx]
        for i in range(1, 5):
            self.assertTrue(allocation[0, 0, i] < allocation[0, 0, i - 1])

    def test_update_temporal_link_and_precedence(self):
        pass

    def test_update_usage(self):
        pass


if __name__ == '__main__':
    unittest.main()

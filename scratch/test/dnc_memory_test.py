"""Test for DNC memory and addressing
"""
import unittest

import torch

from scratch.layers.dnc.memory import Memory


class TestDNCMemory(unittest.TestCase):
    """Test for DNC memory and addressing
    """

    def setUp(self):
        pass

    def test_reset(self):
        memory = Memory((3, 4), 5, 6)
        memory.memory += 1
        memory.temporal_link += 2
        memory.usage += 3
        memory.allocation_weight += 4
        memory.precedence += 5
        memory.read_weights += 6
        memory.reset()
        self.assertEqual(torch.sum(memory.memory != 0), 0)
        self.assertEqual(torch.sum(memory.temporal_link != 0), 0)
        self.assertEqual(torch.sum(memory.usage != 0), 0)
        self.assertEqual(torch.sum(memory.allocation_weight != 0), 0)
        self.assertEqual(torch.sum(memory.precedence != 0), 0)
        self.assertEqual(torch.sum(memory.read_weights != 0), 0)

    def test_write_addressing(self):
        memory


if __name__ == '__main__':
    unittest.main()

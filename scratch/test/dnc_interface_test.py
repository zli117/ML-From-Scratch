"""The test for DNC interface class
"""
import random
import unittest

import torch

from scratch.layers.dnc.interface import Interface


class TestDNCInterface(unittest.TestCase):
    """Test cases for DNC interface object
    """

    def setUp(self):
        batch_size = random.randint(1, 20)
        num_read_heads = random.randint(1, 100)
        cell_width = random.randint(1, 100)
        data_len = (batch_size * ((cell_width * num_read_heads) + 3 * cell_width
                                  + 5 * num_read_heads + 3))
        self.data = torch.arange(
            0, data_len, dtype=torch.float).view(batch_size, -1)
        self.batch_size = batch_size
        self.num_read_heads = num_read_heads
        self.cell_width = cell_width
        self.interface = Interface(self.num_read_heads, self.data,
                                   self.cell_width)

    def test_read_keys(self):
        read_keys = self.interface.read_keys
        self.assertEqual(
            read_keys.shape,
            (self.batch_size, self.num_read_heads, self.cell_width))

    def test_read_strengths(self):
        read_strength = self.interface.read_strength
        self.assertEqual(read_strength.shape,
                         (self.batch_size, self.num_read_heads, 1))

    def test_read_modes(self):
        read_modes = self.interface.read_modes
        self.assertEqual(read_modes.shape,
                         (self.batch_size, self.num_read_heads, 3))

    def test_write_key(self):
        write_key = self.interface.write_key
        self.assertEqual(write_key.shape, (self.batch_size, self.cell_width))

    def test_write_strength(self):
        write_strength = self.interface.write_strength
        self.assertEqual(write_strength.shape, (self.batch_size, 1))

    def test_erase_vector(self):
        erase_vector = self.interface.erase_vector
        self.assertEqual(erase_vector.shape,
                         (self.batch_size, 1, self.cell_width))

    def test_write_vector(self):
        write_vector = self.interface.write_vector
        self.assertEqual(write_vector.shape,
                         (self.batch_size, 1, self.cell_width))

    def test_free_gates(self):
        free_gates = self.interface.free_gates
        self.assertEqual(free_gates.shape,
                         (self.batch_size, self.num_read_heads, 1))

    def test_allocation_gate(self):
        allocation_gate = self.interface.allocation_gate
        self.assertEqual(allocation_gate.shape, (self.batch_size, 1, 1))

    def test_write_gate(self):
        write_gate = self.interface.write_gate
        self.assertEqual(write_gate.shape, (self.batch_size, 1, 1))


if __name__ == '__main__':
    unittest.main()
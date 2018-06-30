"""The test for DNC interface class
"""
import random
import unittest

import torch
from torch.nn import functional

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

    def helper(self,
               start,
               span,
               batch,
               sigmoid_activate=False,
               read_modes=False):
        first_batch = batch[0].view(1, -1)
        target = torch.arange(start, start + span)
        if sigmoid_activate:
            target = functional.sigmoid(target)
        if read_modes:
            target = target.view(-1, 3)
            target = functional.softmax(target, dim=-1)
            target = target.view(1, -1)
        self.assertEqual(torch.sum(first_batch != target), 0)

    def test_read_keys(self):
        read_keys = self.interface.read_keys
        self.assertEqual(
            read_keys.shape,
            (self.batch_size, self.num_read_heads, self.cell_width))
        start = 0
        span = self.num_read_heads * self.cell_width
        self.helper(start, span, read_keys)

    def test_read_strengths(self):
        read_strength = self.interface.read_strength
        self.assertEqual(read_strength.shape,
                         (self.batch_size, self.num_read_heads, 1))
        start = self.num_read_heads * self.cell_width
        span = self.num_read_heads
        self.helper(start, span, read_strength)

    def test_write_key(self):
        write_key = self.interface.write_key
        self.assertEqual(write_key.shape, (self.batch_size, self.cell_width))
        start = self.num_read_heads * (self.cell_width + 1)
        span = self.cell_width
        self.helper(start, span, write_key)

    def test_write_strength(self):
        write_strength = self.interface.write_strength
        self.assertEqual(write_strength.shape, (self.batch_size, 1))
        start = self.num_read_heads * (self.cell_width + 1) + self.cell_width
        span = 1
        self.helper(start, span, write_strength)

    def test_erase_vector(self):
        erase_vector = self.interface.erase_vector
        self.assertEqual(erase_vector.shape,
                         (self.batch_size, 1, self.cell_width))
        start = (
            self.num_read_heads * (self.cell_width + 1) + self.cell_width + 1)
        span = self.cell_width
        self.helper(start, span, erase_vector)

    def test_write_vector(self):
        write_vector = self.interface.write_vector
        self.assertEqual(write_vector.shape,
                         (self.batch_size, 1, self.cell_width))
        start = (self.num_read_heads * (self.cell_width + 1) +
                 self.cell_width * 2 + 1)
        span = self.cell_width
        self.helper(start, span, write_vector)

    def test_free_gates(self):
        free_gates = self.interface.free_gates
        self.assertEqual(free_gates.shape,
                         (self.batch_size, self.num_read_heads, 1))
        start = (self.num_read_heads * (self.cell_width + 1) +
                 self.cell_width * 3 + 1)
        span = self.num_read_heads
        self.helper(start, span, free_gates, sigmoid_activate=True)

    def test_allocation_gate(self):
        allocation_gate = self.interface.allocation_gate
        self.assertEqual(allocation_gate.shape, (self.batch_size, 1, 1))
        start = (self.num_read_heads * (self.cell_width + 2) +
                 self.cell_width * 3 + 1)
        span = 1
        self.helper(start, span, allocation_gate, sigmoid_activate=True)

    def test_write_gate(self):
        write_gate = self.interface.write_gate
        self.assertEqual(write_gate.shape, (self.batch_size, 1, 1))
        start = (self.num_read_heads * (self.cell_width + 2) +
                 self.cell_width * 3 + 2)
        span = 1
        self.helper(start, span, write_gate, sigmoid_activate=True)

    def test_read_modes(self):
        read_modes = self.interface.read_modes
        self.assertEqual(read_modes.shape,
                         (self.batch_size, self.num_read_heads, 3))
        start = (self.num_read_heads * (self.cell_width + 2) +
                 self.cell_width * 3 + 3)
        span = self.num_read_heads * 3
        self.helper(start, span, read_modes, read_modes=True)


if __name__ == '__main__':
    unittest.main()
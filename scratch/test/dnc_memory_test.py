"""Test for DNC memory and addressing
"""
import unittest

import numpy as np
import torch
from torch.autograd import Variable

from scratch.layers.dnc.interface import Interface, InterfaceBuilder
from scratch.layers.dnc.memory import DNCState, Memory


class TestDNCMemory(unittest.TestCase):
    """Test for DNC memory and addressing
    """

    def setUp(self):
        self.read_heads = 2
        self.memory = Memory()

    def test_update_write_weight(self):
        memory = torch.DoubleTensor(
            [[1, -1, -1, 1], [4, 5, 6, 7], [-1, 1, 1, -1], [12, 16, 14, 10],
             [1, -1, 1, -1], [-1, 1, -1, 1]]).view(2, 3, 4)
        usage = torch.DoubleTensor([[0, 0, 1], [1, 0, 0]]).view(2, 1, 3)
        state = DNCState(
            memory=memory,
            usage=usage,
            temporal_link=None,
            precedence=None,
            read_weights=None,
            write_weight=None)
        write_keys = torch.DoubleTensor([[4, 5, 6, 7], [12, 16, 14, 10]]).view(
            2, 1, 4)
        write_strength = torch.DoubleTensor([1, 0.5]).view(2, 1, 1)
        allocation_gate = torch.DoubleTensor([1, 0]).view(2, 1, 1)
        write_gate = torch.DoubleTensor([1, 1]).view(2, 1, 1)
        interface = Interface(
            write_key=write_keys,
            write_strength=write_strength,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            read_keys=None,
            read_strength=None,
            read_modes=None,
            erase_vector=None,
            write_vector=None,
            free_gates=None)
        new_state = self.memory._update_write_weight(interface, state)
        write_weight = new_state.write_weight
        # The first batch only uses allocation weight
        self.assertAlmostEqual(write_weight[0, 0, 0], 1)
        self.assertAlmostEqual(write_weight[0, 0, 1], 0)
        self.assertAlmostEqual(write_weight[0, 0, 2], 0)
        # The second batch only uses content weight
        self.assertTrue(write_weight[1, 0, 0] > write_weight[1, 0, 1])
        self.assertTrue(write_weight[1, 0, 0] > write_weight[1, 0, 2])
        self.assertTrue(write_weight[1, 0, 1] == write_weight[1, 0, 2])

    def test_get_content_addressing(self):
        memory = torch.DoubleTensor(
            [[1, -1, -1, 1], [4, 5, 6, 7], [-1, 1, 1, -1], [12, 16, 14, 10],
             [1, -1, 1, -1], [-1, 1, -1, 1]]).view(2, 3, 4)
        keys = torch.DoubleTensor([[4, 5, 6, 7], [4, 5, 6, 7], [12, 16, 14, 10],
                                   [12, 16, 14, 12]]).view(
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
        memory = torch.DoubleTensor([[1, -1, -1, 1], [4, 5, 6, 7],
                                     [-1, 1, 1, -1]]).view(1, 3, 4)
        link = torch.DoubleTensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]).view(
            1, 3, 3)
        prev_read_weight = torch.DoubleTensor([[1, 0, 0], [0, 0, 1],
                                               [0, 1, 0]]).view(1, 3, 3)
        read_keys = torch.DoubleTensor([[1, -1, -1, 1], [1, -1, -1, 1],
                                        [4, 5, 6, 7]]).view(1, 3, 4)
        read_strength = torch.DoubleTensor([1, 1, 1]).view(1, 3, 1)
        read_modes = torch.DoubleTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).view(
            1, 3, 3)
        interface = Interface(
            read_keys=read_keys,
            read_strength=read_strength,
            read_modes=read_modes,
            write_key=None,
            write_strength=None,
            erase_vector=None,
            write_vector=None,
            free_gates=None,
            allocation_gate=None,
            write_gate=None)
        state = DNCState(
            memory=memory,
            temporal_link=link,
            read_weights=prev_read_weight,
            usage=None,
            precedence=None,
            write_weight=None)
        new_state = self.memory._update_read_weight(interface, state)
        new_read_weights = new_state.read_weights.numpy()
        expected_read_weight = np.array([[0, 0, 1], [1, 0, 0],
                                         [0.21194156, 0.57611688,
                                          0.21194156]]).reshape(1, 3, 3)
        self.assertTrue(np.allclose(new_read_weights, expected_read_weight))

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
        link = torch.DoubleTensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]).view(
            1, 3, 3)
        write_weight = torch.DoubleTensor([0, 0.5, 0.3]).view(1, 1, 3)
        precedence = torch.DoubleTensor([0.1, 0.3, 0.6]).view(1, 1, 3)
        state = DNCState(
            write_weight=write_weight,
            temporal_link=link,
            precedence=precedence,
            memory=None,
            usage=None,
            read_weights=None)
        new_state = self.memory._update_temporal_link_and_precedence(
            state, torch.transpose(write_weight, 1, 2))
        new_precedence = new_state.precedence.numpy()
        new_link = new_state.temporal_link.numpy()
        expected_precedence = (precedence * 0.2 + write_weight).numpy()
        self.assertTrue(np.allclose(expected_precedence, new_precedence))
        expected_link = np.array(
            [[0, 0, 0], [0.05, 0, 0.3], [0.73, 0.09, 0]],
            dtype=np.double).reshape(1, 3, 3)
        self.assertTrue(np.allclose(expected_link, new_link))

    def test_update_usage(self):
        read_weights = torch.DoubleTensor([[0.1, 0.2, 0.3],
                                           [0.3, 0.4, 0.3]]).view(1, 2, 3)
        usage = torch.DoubleTensor([0.1, 0.2, 0.3]).view(1, 1, 3)
        write_weight = torch.DoubleTensor([0.3, 0.4, 0.1]).view(1, 1, 3)
        free_gates = torch.DoubleTensor([0.5, 0.7]).view(1, 2, 1)
        interface = Interface(
            free_gates=free_gates,
            read_keys=None,
            read_strength=None,
            read_modes=None,
            write_key=None,
            write_strength=None,
            erase_vector=None,
            write_vector=None,
            allocation_gate=None,
            write_gate=None)
        state = DNCState(
            usage=usage,
            read_weights=read_weights,
            write_weight=write_weight,
            memory=None,
            temporal_link=None,
            precedence=None)
        new_state = self.memory._update_usage(interface, state)
        new_usage = new_state.usage.numpy()
        self.assertTrue(
            np.allclose(np.array([0.277685, 0.33696, 0.248455]), new_usage))

    def test_forward(self):
        memory = Variable(torch.DoubleTensor(2, 3, 4).fill_(0))
        usage = Variable(torch.DoubleTensor(2, 1, 3).fill_(0))
        read_weights = Variable(torch.DoubleTensor(2, 2, 3).fill_(0))
        write_weight = Variable(torch.DoubleTensor(2, 1, 3).fill_(0))
        temporal_link = Variable(torch.DoubleTensor(2, 3, 3).fill_(0))
        precedence = Variable(torch.DoubleTensor(2, 1, 3).fill_(0))
        state = DNCState(
            usage=usage,
            read_weights=read_weights,
            write_weight=write_weight,
            memory=memory,
            temporal_link=temporal_link,
            precedence=precedence)
        data_len = 2 * ((4 * 2) + 3 * 4 + 5 * 2 + 3)
        interface_vector = Variable(torch.DoubleTensor(data_len).fill_(0)).view(
            2, -1)
        interface = InterfaceBuilder(2, 4).get_interface(interface_vector)
        interface = interface._replace(
            write_vector=interface.write_vector + 1,
            allocation_gate=interface.allocation_gate + 0.5)
        read_val, new_state = self.memory(interface, state)
        read_val, new_state = self.memory(interface, new_state)
        read_val, new_state = self.memory(interface, new_state)
        self.assertEqual(new_state.temporal_link[0, 1, 0], 0.25)
        self.assertEqual(new_state.temporal_link[1, 1, 0], 0.25)
        memory = new_state.memory.numpy()
        self.assertTrue(np.allclose(memory, 0.5))

if __name__ == '__main__':
    unittest.main()

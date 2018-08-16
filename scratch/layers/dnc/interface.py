"""
The interface data structure
"""
import collections

import torch
from torch.nn import functional

# Size of each element:
# read_keys: (B, C, W)
# read_strength: (B, C, 1)
# read_modes: (B, C, 3)
# write_key: (B, 1, W)
# write_strength: (B, 1, 1)
# erase_vector: (B, 1, W)
# write_vector: (B, 1, W)
# free_gates: (B, C, 1)
# allocation_gate: (B, 1, 1)
# write_gate: (B, 1, 1)
Interface = collections.namedtuple(
    "Interface", ("read_keys", "read_strength", "read_modes", "write_key",
                  "write_strength", "erase_vector", "write_vector",
                  "free_gates", "allocation_gate", "write_gate"))


class InterfaceBuilder:
    """Builds the interface data structure
    Takes an interface vector emitted from the controller and splits up into
    each specific key.
    """

    def __init__(self, num_read_heads, cell_width):
        self.num_read_heads = num_read_heads
        self.cell_width = cell_width

    def get_interface(self, vector):
        """Parse the vector to interface

        Args:
            vector (torch.Tensor): The interface vector emitted from the
                                   controller

        Returns:
            Interface: The interface tuple
        """

        interface = Interface(
            read_keys=self._get_read_keys(vector),
            read_strength=self._get_read_strengths(vector),
            read_modes=self._get_read_modes(vector),
            write_key=self._get_write_key(vector),
            write_strength=self._get_write_strength(vector),
            erase_vector=self._get_erase_vector(vector),
            write_vector=self._get_write_vector(vector),
            free_gates=self._get_r_free_gates(vector),
            allocation_gate=self._get_allocation_gate(vector),
            write_gate=self._get_write_gate(vector))
        return interface

    def _get_read_keys(self, interface):
        keys = interface[:, :self.num_read_heads * self.cell_width]
        return keys.view(-1, self.num_read_heads, self.cell_width)

    def _get_read_strengths(self, interface):
        start = self.num_read_heads * self.cell_width
        strengths = interface[:, start:start + self.num_read_heads]
        return strengths.unsqueeze(dim=2)

    def _get_write_key(self, interface):
        start = self.num_read_heads * (self.cell_width + 1)
        key = interface[:, start:(start + self.cell_width)]
        return key.view(-1, 1, self.cell_width)

    def _get_write_strength(self, interface):
        strength = interface[:, self.num_read_heads * (self.cell_width + 1) +
                             self.cell_width].unsqueeze(dim=1)
        return strength.view(-1, 1, 1)

    def _get_erase_vector(self, interface):
        start = (
            self.num_read_heads * (self.cell_width + 1) + self.cell_width + 1)
        erase_vector = interface[:, start:(start + self.cell_width)]
        return erase_vector.unsqueeze(dim=1)

    def _get_write_vector(self, interface):
        start = (self.num_read_heads * (self.cell_width + 1) +
                 self.cell_width * 2 + 1)
        write_vector = interface[:, start:(start + self.cell_width)]
        return write_vector.unsqueeze(dim=1)

    def _get_r_free_gates(self, interface):
        start = (self.num_read_heads * (self.cell_width + 1) +
                 self.cell_width * 3 + 1)
        gates = interface[:, start:(start + self.num_read_heads)]
        return torch.sigmoid(gates).unsqueeze(dim=2)

    def _get_allocation_gate(self, interface):
        start = (self.num_read_heads * (self.cell_width + 2) +
                 self.cell_width * 3 + 1)
        gate = interface[:, start].unsqueeze(dim=1)
        return torch.sigmoid(gate).unsqueeze(dim=2)

    def _get_write_gate(self, interface):
        start = (self.num_read_heads * (self.cell_width + 2) +
                 self.cell_width * 3 + 2)
        gate = interface[:, start].unsqueeze(dim=1)
        return torch.sigmoid(gate).unsqueeze(dim=2)

    def _get_read_modes(self, interface):
        modes = interface[:, -3 * self.num_read_heads:]
        modes = modes.view(-1, self.num_read_heads, 3)
        return functional.softmax(modes, dim=2)

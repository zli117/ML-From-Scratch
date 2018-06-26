"""
The interface data structure
"""
import torch.nn.functional as functional


class Interface:
    """The interface data structure
    Takes an interface vector emitted from the controller and splits up into
    each specific key.
    """
    def __init__(self, num_read_heads, interface, cell_width):
        self.num_read_heads = num_read_heads
        self.cell_width = cell_width
        self.read_keys = self.get_read_keys_(interface) # (B, #RH, W)
        self.read_strength = self.get_read_strengths_(interface) # (B, #RH, 1)
        self.read_modes = self.get_read_modes_(interface) # (B, #RH, 3)
        self.write_key = self.get_write_key_(interface) # (B, W)
        self.write_strength = self.get_write_strength_(interface) # (B, 1)
        self.erase_vector = self.get_erase_vector_(interface) # (B, 1, W)
        self.write_vector = self.get_write_vector_(interface) # (B, 1, W)
        self.free_gates = self.get_r_free_gates(interface) # (B, #RH, 1)
        self.allocation_gate = self.get_allocation_gate(interface) # (B, 1, 1)
        self.write_gate = self.get_write_gate(interface) # (B, 1, 1)

    def get_read_keys_(self, interface):
        keys = interface[:, :self.num_read_heads * self.cell_width]
        return keys.unsqueeze(dim=0)

    def get_read_strengths_(self, interface):
        start = self.num_read_heads * self.cell_width
        strengths = interface[:, start:start + self.num_read_heads]
        return strengths.unsqueeze(dim=2)

    def get_write_key_(self, interface):
        start = self.num_read_heads * (self.cell_width + 1)
        return interface[:, start:(start + self.cell_width)]

    def get_write_strength_(self, interface):
        return interface[:, self.num_read_heads * (self.cell_width + 1) +
                         self.cell_width]

    def get_erase_vector_(self, interface):
        start = self.num_read_heads * (
            self.cell_width + 1) + self.cell_width + 1
        erase_vector = interface[:, start:(start + self.cell_width)]
        return erase_vector.unsqueeze(dim=1)

    def get_write_vector_(self, interface):
        start = self.num_read_heads * (
            self.cell_width + 1) + self.cell_width * 2 + 1
        write_vector = interface[:, start:(start + self.cell_width)]
        return write_vector.unsqueeze(dim=1)

    def get_r_free_gates(self, interface):
        start = self.num_read_heads * (
            self.cell_width + 1) + self.cell_width * 3 + 1
        gates = interface[:, start:(start + self.num_read_heads)]
        return functional.sigmoid(gates).unsqueeze(dim=2)

    def get_allocation_gate(self, interface):
        start = self.num_read_heads * (
            self.cell_width + 2) + self.cell_width * 3 + 1
        gate = interface[:, start]
        return functional.sigmoid(gate).unsqueeze(dim=2)

    def get_write_gate(self, interface):
        start = self.num_read_heads * (
            self.cell_width + 2) + self.cell_width * 3 + 2
        gate = interface[:, start]
        return functional.sigmoid(gate).unsqueeze(dim=2)

    def get_read_modes_(self, interface):
        """Extracts the read modes from the interface for all batches

        Args:
            interface (torch.Tensor): interface vector

        Returns:
            torch.Tensor: shape: (B, #RH, 3)
        """
        modes = interface[:, -3 * self.num_read_heads]
        return modes.view(-1, self.num_read_heads, 3)

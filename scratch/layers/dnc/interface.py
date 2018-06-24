"""
The interface data structure
"""
import torch.nn.functional as functional


class Interface:
    """The interface data structure
    Takes an interface vector emitted from the controller and splits up into
    each specific key.
    """
    def __init__(self, num_read_heads, interface, w):
        self.num_read_heads_ = num_read_heads
        self.memory_width = w
        self.read_keys = []
        self.read_strength = []
        self.read_modes = []
        for i in range(num_read_heads):
            self.read_keys.append(self.get_read_key_i_(i, interface))
            self.read_strength.append(self.get_read_strength_i_(i, interface))
            self.read_modes.append(self.get_read_modes_i_(i, interface))
        self.write_key = self.get_write_key_(interface)
        self.write_strength = self.get_write_strength_(interface)
        self.erase_vector = self.get_erase_vector_(interface)
        self.write_vector = self.get_write_vector_(interface)
        self.free_gates = self.get_r_free_gates(interface)
        self.allocation_gate = self.get_allocation_gate(interface)
        self.write_gate = self.get_write_gate(interface)

    def get_read_key_i_(self, i, interface):
        return interface[:, i * self.memory_width:(i + 1) * self.memory_width]

    def get_read_strength_i_(self, i, interface):
        return interface[:, self.num_read_heads_ * self.memory_width + i]

    def get_write_key_(self, interface):
        start = self.num_read_heads_ * (self.memory_width + 1)
        return interface[:, start:(start + self.memory_width)]

    def get_write_strength_(self, interface):
        return interface[:, self.num_read_heads_ * (self.memory_width + 1) +
                         self.memory_width]

    def get_erase_vector_(self, interface):
        start = self.num_read_heads_ * (
            self.memory_width + 1) + self.memory_width + 1
        return interface[:, start:(start + self.memory_width)]

    def get_write_vector_(self, interface):
        start = self.num_read_heads_ * (
            self.memory_width + 1) + self.memory_width * 2 + 1
        return interface[:, start:(start + self.memory_width)]

    def get_r_free_gates(self, interface):
        start = self.num_read_heads_ * (
            self.memory_width + 1) + self.memory_width * 3 + 1
        gates = interface[:, start:(start + self.num_read_heads_)]
        return functional.sigmoid(gates)

    def get_allocation_gate(self, interface):
        start = self.num_read_heads_ * (
            self.memory_width + 2) + self.memory_width * 3 + 1
        gate = interface[:, start]
        return functional.sigmoid(gate)

    def get_write_gate(self, interface):
        start = self.num_read_heads_ * (
            self.memory_width + 2) + self.memory_width * 3 + 2
        gate = interface[:, start]
        return functional.sigmoid(gate)

    def get_read_modes_i_(self, interface, i):
        start = self.num_read_heads_ * (
            self.memory_width + 2) + self.memory_width * 3 + 3 * i + 3
        gates = interface[:, start:(start + 3)]
        return functional.sigmoid(gates)

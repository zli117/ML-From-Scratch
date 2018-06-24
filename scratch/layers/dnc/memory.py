"""
The memory module of DNC
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Module, Parameter


class Memory(Module):
    """The memory module for DNC
    """

    def __init__(self, memory_size, num_read_heads):
        """Create a memory module.

        The memory module will be responsible for read write to the memory
        (The addressing, allocation, temporal link).

        Args:
            memory_size (Tuple[int, int]): The size of the memory
            num_read_heads (int): The number of read heads
        """
        super().__init__()
        self.memory = Parameter(torch.Tensor(memory_size), requires_grad=False)
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.usage = Parameter(torch.zeros(memory_size[0], 1))
        self.reset_memory()

    def reset_memory(self):
        """Zeros out the memory and usage vector
        """
        self.memory.data.zero_()
        self.usage.data.zero_()

    def addressing_(self, key, weight):
        """Compute the content based addressing as described in the paper

        Args:
            key (torch.Tensor): The lookup key. Should be a tensor with shape
                (1, N)
            weight (torch.Tensor): The read weight or write weight. Should be a
                tensor with shape (1, 1)

        Returns:
            torch.Tensor: The probability for each row to be used.
        """

        cos_similarity = nn.CosineSimilarity()
        similarity = cos_similarity(key, self.memory)
        return functional.softmax(similarity * weight)

    def get_allocation_(self):
        """Computes the allocation weightings as described in the paper

        Returns:
            torch.Tensor: The allocation weight for each memory entry (row in
                the memory matrix)
        """

        _, idx = torch.sort(self.usage)
        allocation = torch.ones_like(self.usage)
        multiply = 1
        for i in idx:
            allocation[i][0] = (1 - self.usage[i][0]) * multiply
            multiply *= self.usage[i, 0]
        return allocation

    def forward(self, interface):
        """Performs one read and write cycle

        The following steps are deduced from the formulas in the paper:
            1. Computes the allocation weights
            2. Computes the weight weights using allocation weights and
               content weights.
            3. Write to the memory matrix
            4. Update the temporal link matrix
            5. Compute forward and backward weights
            6. Compute the read weight
            7. Perform read
            8. Update usage vector

        Note that we first perform the write then perform the read. This is
        because in the paper read weight depends on the current temporal linkage
        matrix which can only be updated after write.

        Another notice is that the interface vectors here are all row vectors,
        unlike in the paper

        Args:
            interface (Interface): The interface emitted from the controller
        """
        allocation_weight = self.get_allocation_()
        write_content_address = self.addressing_(interface.write_key,
                                                 interface.write_strength)
        write_weights = interface.write_gate * (
            interface.allocation_gate * allocation_weight +
            (1 - interface.allocation_gate) * write_content_address)

        # Perform the write

        self.memory = self.memory * (1 - )

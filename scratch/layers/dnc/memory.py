"""
The memory module of DNC
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Module, Parameter


class Memory(Module):
    def __init__(self, memory_size, num_read_heads):
        """
        Create a memory module. The memory module will be responsible for
        read write to the memory (The addressing, allocation, temporal link).
        :param memory_size: The size of the memory (N, W)
        :param num_read_heads: The number of read heads
        """
        super().__init__()
        self.memory = Parameter(torch.Tensor(memory_size), requires_grad=False)
        self.ones = Parameter(torch.ones(memory_size), requires_grad=False)
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.usage = Parameter(torch.zeros(memory_size[0], 1))
        self.reset_memory()

    def reset_memory(self):
        self.memory.data.zero_()
        self.usage.data.zero_()

    def addressing_(self, key, weight):
        """
        Compute the content based addressing. The result will be
        :param key: The lookup key. Should be a tensor with shape (1, N)
        :param weight: A scalar
        :return: The probability for each row to be used.
        """
        cos_similarity = nn.CosineSimilarity()
        similarity = cos_similarity(key, self.memory)
        return functional.softmax(similarity * weight)

    def get_allocation_(self):
        _, idx = torch.sort(self.usage)
        allocation = torch.ones_like(self.usage)
        multiply = 1
        for i in idx:
            allocation[i][0] = (1 - self.usage[i][0]) * multiply
            multiply *= self.usage[i, 0]
        return allocation

    def forward(self, interface):
        write_content_address = self.addressing_(interface.write_key,
                                                 interface.write_strength)


"""
The implementation of the differential neural computer
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Module
from .interface import Interface


class Controller(Module):
    def __init__(self, memory_size, input_size, output_size, num_read_heads,
                 controller=nn.LSTMCell):
        super().__init__()
        n, w = memory_size
        # TODO: Add support for multiple write heads
        self.controller = controller(
            num_read_heads * w + input_size,
            (num_read_heads + 3) * w + 5 * num_read_heads + 3 + output_size)
        self.n = n
        self.w = w
        self.input_size = input_size
        self.output_size = output_size
        self.num_read_heads = num_read_heads

    def split_vectors_(self, output):
        interface_size = self.num_read_heads * (self.w_ + 5) + self.w_ * 3 + 3
        return output[:interface_size], output[interface_size:]

    def forward(self, x, prev_read_results, prev_states):
        """
        Perform one step.
        :param x: The input vector
        :param prev_read_results: A list of reading from the read heads. Each
                                  read vector should be a row vector.
        :param prev_states: The previous states for the lstm
        :return: The interface, the output vector and the memory state of the
                 lstm.
        """
        prev_read_results.append(x)
        concat_input = torch.cat(prev_read_results, dim=0)
        (h, c) = self.controller(concat_input, prev_states)
        interface, output_vector = self.split_vectors_(h)
        interface = Interface(self.num_read_heads, interface, self.w)
        return interface, output_vector, c

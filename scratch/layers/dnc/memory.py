"""
The memory module of DNC
"""
import collections

import torch
from torch import nn
from torch.nn import Module, functional
from torch.autograd import Variable

# Size of each field:
# memory: (B, N, W)
# temporal_link: (B, N, N)
# usage: (B, 1, N)
# precedence: (B, 1, N)
# read_weights: (B, R, N)
# write_weight: (B, 1, N)
DNCState = collections.namedtuple(
    "DNCState", ("memory", "temporal_link", "usage", "precedence",
                 "read_weights", "write_weight"))


class Memory(Module):
    """The memory function.

    All the tensor should be batch major

    Does not have any internal state.
    """

    def __init__(self):
        """Create a memory module.

        The memory module will be responsible for perform read write to the
        memory (The addressing, allocation, temporal link).

        Args:
            memory_size (Tuple[int, int]): The size of the memory
            num_read_heads (int): The number of read heads
        """

        super().__init__()
        self.cos_similarity = nn.CosineSimilarity()

    def _update_write_weight(self, interface, state):
        """Compute the write weight

        Args:
            interface (Interface): The controller interface
            state (DNCState): Current state

        Returns:
            DNCState: With write weight updated
        """

        allocation_weight = self._get_allocation_weight(state.usage)

        write_content = self._get_content_addressing(
            interface.write_key, interface.write_strength, state.memory)
        write_weight = (interface.allocation_gate *
                        (allocation_weight - write_content) + write_content)
        write_weight *= interface.write_gate
        return state._replace(write_weight=write_weight)

    def _get_content_addressing(self, keys, strength, memory):
        """Compute the content addressing weight for each cell

        Args:
            keys (torch.Tensor): The keys. Should have shape (B, C, W)
            strength (torch.Tensor): The strength. Should have shape (B, C, 1)

        Returns:
            torch.Tensor: The probability for each row to be read by each read
                head. The shape is (B, C, N)
        """

        batch_size = memory.shape[0]
        num_cells = memory.shape[1]
        num_channels = keys.shape[1]
        keys = keys.unsqueeze(dim=2)
        addressing = []

        # Loop needed here since PyTorch doesn't have batch cos_similarity

        for i in range(batch_size):
            batch_key = keys[i]
            batch_strength = strength[i]
            similarities = [
                self.cos_similarity(batch_key[j], memory[i])
                for j in range(num_channels)
            ]
            similarities = torch.stack(
                similarities, dim=0).view(num_channels, num_cells)
            softmax = functional.softmax(similarities * batch_strength, dim=1)
            addressing.append(softmax)
        return torch.stack(addressing, dim=0)

    def _update_read_weight(self, interface, state):
        """Compute the read weight

        Args:
            interface (Interface): The interface emitted from the controller
            state (DNCState): The state

        Returns:
            DNCState: State with read weights updated
        """

        transpose_temporal_link = torch.transpose(state.temporal_link, 1, 2)
        forward_weights = torch.bmm(state.read_weights, transpose_temporal_link)
        backward_weights = torch.bmm(state.read_weights, state.temporal_link)
        content_weights = self._get_content_addressing(
            interface.read_keys, interface.read_strength, state.memory)
        forward_modes = interface.read_modes[:, :, 0].unsqueeze(dim=2)
        backward_modes = interface.read_modes[:, :, 1].unsqueeze(dim=2)
        content_modes = interface.read_modes[:, :, 2].unsqueeze(dim=2)
        read_weights = (
            forward_weights * forward_modes + backward_weights * backward_modes
            + content_weights * content_modes)
        return state._replace(read_weights=read_weights)

    def _get_allocation_weight(self, usage):
        """Compute the allocation weight from current usage

        Args:
            usage (torch.Tensor): current usage (B, 1, N)

        Returns:
            torch.Tensor: The allocation weight for each memory entry (row in
                          the memory matrix). Shape: (B, 1, N)
        """

        batch_size = usage.shape[0]
        sorted_usage, idx = torch.sort(usage, dim=2)
        _, rev_idx = torch.sort(idx, dim=2)
        ones = Variable(sorted_usage.data.new(batch_size, 1, 1).fill_(1))
        acc_prod_usage = torch.cumprod(
            torch.cat((ones, sorted_usage), dim=2), dim=2)[:, :, :-1]
        sorted_allocation = (1 - sorted_usage) * acc_prod_usage
        return torch.gather(sorted_allocation, 2, rev_idx)

    def _update_temporal_link_and_precedence(self, state,
                                             transpose_write_weight):
        """Compute the new temporal link and precedence

        Args:
            state (DNCState): The current state
            transpose_write_weight ([type]): transposed write weight. Shape:
                                             (B, N, 1)

        Returns:
            DNCState: State with temporal link and precedence updated
        """

        num_cells = state.write_weight.shape[2]
        grid_sum = (transpose_write_weight.repeat(1, 1, num_cells) +
                    state.write_weight.repeat(1, num_cells, 1))
        grid_subtract = 1 - grid_sum
        temporal_link = (grid_subtract * state.temporal_link + torch.bmm(
            transpose_write_weight, state.precedence))
        mask = 1 - torch.eye(
            num_cells, device=temporal_link.device, dtype=temporal_link.dtype)
        temporal_link *= mask.expand_as(temporal_link)
        precedence = ((1 - torch.sum(state.write_weight)) * state.precedence +
                      state.write_weight)
        return state._replace(
            temporal_link=temporal_link, precedence=precedence)

    def _update_usage(self, interface, state):
        """Update the usage vector in state

        Args:
            interface (Interface): The interface of from controller
            state (DNCState): The state

        Returns:
            DNCState: State with usage updated
        """

        prev_read_weights = state.read_weights
        retention_vector = torch.prod(
            1 - interface.free_gates * prev_read_weights,
            dim=1).unsqueeze(dim=1)
        usage = state.usage
        usage = ((usage + state.write_weight - usage * state.write_weight) *
                 retention_vector)
        return state._replace(usage=usage)

    def forward(self, interface, state):
        """Perform one read write on the memory

        Computes the weights from the interface emitted from the controller.

        Args:
            interface (Interface): The interface from controller
            state (DNCState): The current state

        Returns:
            Tuple(torch.Tensor, torch.Tensor): The read result. Shape:
                                               (Batch, #RH, W), and the state
        """
        state = self._update_write_weight(interface, state)
        write_vector = interface.write_vector
        erase_vector = interface.erase_vector
        transpose_write_weight = torch.transpose(state.write_weight, 1, 2)

        memory = state.memory
        memory *= 1 - torch.bmm(transpose_write_weight, erase_vector)
        memory += torch.bmm(transpose_write_weight, write_vector)
        state = state._replace(memory=memory)

        state = self._update_temporal_link_and_precedence(
            state, transpose_write_weight)

        state = self._update_read_weight(interface, state)
        read_val = torch.bmm(state.read_weights, state.memory)

        state = self._update_usage(interface, state)

        return read_val, state

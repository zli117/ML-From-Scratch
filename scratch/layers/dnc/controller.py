"""
The implementation of the differential neural computer
"""
import collections

import torch
from torch.autograd import Variable
from torch.nn import Module

from .interface import InterfaceBuilder
from .memory import Memory, MemoryState

DNCState = collections.namedtuple(
    "DNCState", ("memory_state", "controller_hidden", "read_result"))


class Controller(Module):
    """The controller wrapper for DNC

    Wraps around the logic for updating the memory and creating the controller
    network.
    """

    def __init__(self, memory_size, num_read_heads, controller_factory,
                 input_dim, output_dim, **kwargs):
        """The controller wrapper for DNC

        Args:
            memory_size (Tuple(int, int)): The number of cells and size of a
                cell
            num_read_heads (int): Number of read heads
            controller_factory (function): A factory for creating the
                controller. The first two arguments are: The readout dimension
                and the interface vector dimension. The interface will come
                before the output in controller out vector. Return value should
                be a tuple of the actual controller and the initial hidden state
                or None if there is no hidden state.
            input_dim (int): The dimension of input
            output_dim (int): The dimension of output
        """

        super().__init__()
        num_cells, cell_width = memory_size
        self.interface_size = (
            (num_read_heads + 3) * cell_width + 5 * num_read_heads + 3)
        self.controller, self.controller_init_hidden = controller_factory(
            num_read_heads * cell_width + input_dim,
            self.interface_size + output_dim, **kwargs)
        self.interface_builder = InterfaceBuilder(num_read_heads, cell_width)
        self.memory = Memory()
        self.num_cells = num_cells
        self.cell_width = cell_width
        self.num_read_heads = num_read_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _split_controller_output(self, controller_out):
        interface = controller_out[:, :self.interface_size]
        output = controller_out[:, self.interface_size:]
        return interface, output

    def forward(self, x, state):
        """
        Args:
            x (torch.Tensor): Input for this time step. Batch majored
            state ([type]): [description]

        Returns:
            (Tuple(torch.Tensor, DNCState)): Output for this time step and the
                new state
        """

        batch_size = x.shape[0]
        if state is None:
            memory = Variable(
                x.data.new(batch_size, self.num_cells,
                           self.cell_width).fill_(0))
            temporal_link = Variable(
                x.data.new(batch_size, self.num_cells, self.num_cells).fill_(0))
            usage = Variable(x.data.new(batch_size, 1, self.num_cells))
            precedence = Variable(x.data.new(batch_size, 1, self.num_cells))
            memory_state = MemoryState(
                memory=memory,
                temporal_link=temporal_link,
                usage=usage,
                precedence=precedence,
                read_weights=None,
                write_weight=None)
            read_result = Variable(
                x.data.new(batch_size, self.num_read_heads, self.cell_width))
            state = DNCState(
                memory_state=memory_state,
                controller_hidden=self.controller_init_hidden,
                read_result=read_result)

        controller_input = [
            torch.cat((state.read_result.view(batch_size, -1), x), dim=1)
        ]

        if state.controller_hidden is not None:
            controller_input.append(state.controller_hidden)

        controller_output = self.controller(*controller_input)

        interface_vec, output = self._split_controller_output(
            controller_output[0])

        # The first controller output should be the interface vector
        interface = self.interface_builder(interface_vec)
        read_result, memory_state = self.memory(interface, state.memory_state)

        return output, state._replace(
            memory_state=memory_state,
            controller_hidden=controller_output[1]
            if len(controller_output) > 1 else None,
            read_result=read_result)

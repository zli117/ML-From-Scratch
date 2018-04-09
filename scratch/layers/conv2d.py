"""
Convolution 2d layer
"""
import math

import torch
from torch.nn import Module, Parameter
from torch.nn.functional import pad


class Conv2DLayer(Module):
    """
    Convolution 2d layer
    """
    def __init__(self, input_channels, kernel_shape, num_kernels, initializer,
                 stride=(1, 1), padding=(0, 0), bias=True):
        """
        Create a 2D convolution layer
        The expected input size is (b, c, x, y)
        :param input_channels: How many channels does the input have
        :param kernel_shape: The kernel shape (x, y)
        :param num_kernels: The number of kernels
        :param initializer: The initializer
        :param stride: Stride size (left & right and upper & bottom)
        :param padding: Padding on two directions
                        (left & right and upper & bottom)
        :param bias: Whether use bias
        :param dtype: The dtype of the weights
        """
        super().__init__()
        self.kernels = Parameter(
            torch.Tensor(
                num_kernels,
                (kernel_shape[0]
                 * kernel_shape[1]
                 * input_channels)
            )
        )
        self.bias = Parameter(torch.Tensor(num_kernels, 1).zero_())
        self.has_bias = bias
        self.input_channels = input_channels
        self.stride_r, self.stride_c = stride
        self.padding_r, self.padding_c = padding
        self.kernel_r, self.kernel_c = kernel_shape
        self.num_kernels = num_kernels
        self.initializer = initializer
        self.reset_parameters()

    @staticmethod
    def _compute_out_size(in_size, padding, stride, kernel_size):
        return math.floor((in_size + 2 * padding - kernel_size) / stride + 1)

    def reset_parameters(self):
        self.initializer(self.kernels.data)
        if self.has_bias:
            self.initializer(self.bias.data)

    def _unroll(self, inputs, out_r, out_c):
        if len(inputs.shape) != 4 or inputs.shape[1] != self.input_channels:
            raise RuntimeError('Invalid input shape: %s' % inputs.shape)
        padded = pad(inputs, (self.padding_c, self.padding_c,
                              self.padding_r, self.padding_r))
        batch_num, channel_num, row, col = padded.shape
        kernel_unrolled_shape = self.kernel_r * self.kernel_c
        batches = []
        for b in range(batch_num):
            channels = []
            for ch in range(channel_num):
                channel = []
                for r in range(out_r):
                    for c in range(out_c):
                        cur_block = (
                            padded[b, ch, :, :]
                            .narrow(0, r * self.stride_r, self.kernel_r)
                            .narrow(1, c * self.stride_c, self.kernel_c)
                            .contiguous()
                        ).view(kernel_unrolled_shape, 1)
                        channel.append(cur_block)
                channels.append(torch.cat(channel, dim=1))
            batches.append(torch.cat(channels, dim=0))
        return batches

    def forward(self, inputs):
        batch, _, in_r, in_c = inputs.shape
        out_r = self._compute_out_size(in_r, self.padding_r,
                                       self.stride_r, self.kernel_r)
        out_c = self._compute_out_size(in_c, self.padding_c,
                                       self.stride_c, self.kernel_c)
        # The unrolling approach is not as efficient as direct convolution,
        # but since we are implementing it on top of PyTorch, this could be the
        # most efficient way possible.
        unrolled_batches = self._unroll(inputs, out_r, out_c)
        results = [torch.mm(self.kernels, batch) + self.bias
                   for batch in unrolled_batches]
        result_batches = torch.cat(results, dim=0)
        return result_batches.view(batch, self.num_kernels, out_r, out_c)

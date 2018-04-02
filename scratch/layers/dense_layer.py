"""
Dense layer from scratch
-----
Equivalent to the torch.nn.Linear
"""
import torch
from torch.autograd import Function
from torch.nn import Module, Parameter


class DenseFunction(Function):
    """
    An implementation similar to the one in the doc
    """
    @staticmethod
    def forward(ctx, input_mtx, weight_mtx, bias=None):
        """
        Forward pass for dense function
        :param ctx: The context
        :param input_mtx: The input matrix
        :param weight_mtx: The weight matrix
        :param bias: The bias values. If none then assume no bias. Has to have
                     shape (1, out_size)
        :return: The tensor for next layer
        """
        ctx.save_for_backward(input_mtx, weight_mtx, bias)
        output = torch.mm(input_mtx, weight_mtx)
        if bias is not None:
            output += bias.expand_as(output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass for the dense function
        :param ctx:
        :param output_grad:
        :return:
        """
        # output_grad = output_grad.data
        ipt, weight, bias = ctx.saved_variables
        ipt_grad = torch.mm(output_grad, weight.t())
        weight_grad = torch.mm(ipt.t(), output_grad)
        bias_grad = None
        if bias is not None:
            bias_grad = torch.sum(output_grad, dim=0)
        return ipt_grad, weight_grad, bias_grad


class DenseLayer(Module):
    def __init__(self, in_units, out_units, initializer,
                 bias=True, dtype=torch.FloatTensor):
        super().__init__()
        self.weights = Parameter(torch.Tensor(in_units, out_units).type(dtype))
        self.initializer = initializer
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_units).type(dtype))
        else:
            self.register_parameter('bias', None)
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.initializer(self.weights.data)
        if self.bias is not None:
            self.initializer(self.bias.data)

    def forward(self, ipt_mtx):
        return DenseFunction.apply(ipt_mtx, self.weights, self.bias)

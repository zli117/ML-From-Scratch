"""
A relu layer
"""
import torch
from torch.autograd import Function
from torch.nn import Module


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, ipt):
        ctx.save_for_backward(ipt)
        torch.max(ipt, torch.zeros_like(ipt))

    @staticmethod
    def backward(ctx, *grad_outputs):
        ipt = ctx.saved_variables
        return ipt > 0


class ReluLayer(Module):

    def forward(self, ipt):
        ReluFunction.apply(ipt)

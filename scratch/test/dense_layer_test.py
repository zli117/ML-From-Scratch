"""
Unit test for DenseLayer
"""
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck

from scratch.layers.dense_layer import DenseFunction, DenseLayer

inputs_func = (
    Variable(torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True).cuda(),
    Variable(torch.DoubleTensor(30, 40).uniform_(-1, 1), requires_grad=True).cuda(),
    Variable(torch.DoubleTensor(40).uniform_(-1, 1), requires_grad=True).cuda(),
)

inputs_layer = (
    Variable(torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True).cuda(),
)


if __name__ == '__main__':
    layer = DenseLayer(30, 40, lambda t: torch.nn.init.normal(t, -1, 1),
                       dtype=torch.cuda.DoubleTensor).cuda()
    # TODO: Add verification for validity
    tests = [
        gradcheck(DenseFunction.apply, inputs_func, raise_exception=False),
        gradcheck(layer, inputs_layer, raise_exception=False),
    ]
    for i, result in enumerate(tests, 1):
        print('Test%d: %s' % (i, result))

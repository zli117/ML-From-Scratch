"""
Unit test for DenseLayer
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck

from scratch.layers.dense_layer import DenseFunction, DenseLayer


inputs_func = (
    Variable(torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True),
    Variable(torch.DoubleTensor(30, 40).uniform_(-1, 1), requires_grad=True),
    Variable(torch.DoubleTensor(40).uniform_(-1, 1), requires_grad=True),
)

inputs_layer = (
    Variable(torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True),
)


if __name__ == '__main__':
    layer = DenseLayer(30, 40, lambda *args: np.random.normal(),
                       dtype=torch.DoubleTensor)
    # TODO: Add verification for validity
    tests = [
        gradcheck(DenseFunction.apply, inputs_func, raise_exception=False),
        gradcheck(layer, inputs_layer, raise_exception=False),
    ]
    for i, result in enumerate(tests, 1):
        print('Test%d: %s' % (i, result))

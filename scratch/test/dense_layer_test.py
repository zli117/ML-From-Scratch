"""
Unit test for DenseLayer
"""
import unittest

import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck

from scratch.layers.dense_layer import DenseFunction, DenseLayer


class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.inputs_func = [
            Variable(
                torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True
            ),
            Variable(
                torch.DoubleTensor(30, 40).uniform_(-1, 1), requires_grad=True
            ),
            Variable(
                torch.DoubleTensor(40).uniform_(-1, 1), requires_grad=True
            ),
        ]

        self.inputs_layer = [
            Variable(
                torch.DoubleTensor(20, 30).uniform_(-1, 1), requires_grad=True
            ),
        ]

        self.inputs_func_cuda = [
            Variable(
                ipt.data.type(torch.DoubleTensor).cuda(), requires_grad=True
            ) for ipt in self.inputs_func
        ]

        self.inputs_layer_cuda = [
            Variable(
                ipt.data.type(torch.DoubleTensor).cuda(), requires_grad=True
            ) for ipt in self.inputs_layer
        ]

        self.func = DenseFunction.apply

    def test_func_cpu(self):
        self.assertTrue(
            gradcheck(self.func, self.inputs_func, raise_exception=False)
        )

    def test_func_gpu(self):
        self.assertTrue(
            gradcheck(self.func, self.inputs_func_cuda, raise_exception=False)
        )

    def test_layer_cpu(self):
        layer = DenseLayer(30, 40, lambda t: torch.nn.init.normal(t, -1, 1),
                           dtype=torch.DoubleTensor)
        self.assertTrue(
            gradcheck(layer, self.inputs_layer, raise_exception=False)
        )

    def test_layer_gpu(self):
        layer = DenseLayer(30, 40, lambda t: torch.nn.init.normal(t, -1, 1),
                           dtype=torch.cuda.DoubleTensor)
        self.assertTrue(
            gradcheck(layer, self.inputs_layer_cuda, raise_exception=False)
        )


if __name__ == '__main__':
    unittest.main()

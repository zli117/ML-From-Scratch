"""
Test for convolution 2d layer
"""
import unittest

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Conv2d

from scratch.layers.conv2d import Conv2DLayer


class TestConv2DLayer(unittest.TestCase):
    def setUp(self):
        self.configs = [
            # in channel, out channel, kernel size, stride, padding, bias, \
            # batch size, height, width
            (3, 5, 5, 2, 4, False, 2, 10, 10),
            (5, 2, 7, 2, 4, False, 2, 5, 5),
            (5, 2, 7, 2, 4, False, 2, 15, 3),
            (5, 2, 7, 2, 4, False, 2, 3, 15),
            (5, 5, 3, 4, 2, False, 5, 8, 7),
            (3, 5, 5, 2, 4, True, 2, 10, 10),
            (5, 2, 7, 2, 4, True, 2, 5, 5),
            (5, 2, 7, 2, 4, True, 2, 15, 3),
            (5, 2, 7, 2, 4, True, 2, 3, 15),
            (5, 5, 3, 4, 2, True, 5, 8, 7),
        ]

    def helper_func(self, config_idx):
        (in_ch, out_ch, k_size, stride, padding, has_bias,
         batch_size, height, width) = self.configs[config_idx]

        torch_conv2d = Conv2d(
            in_ch, out_ch, k_size, stride=stride, padding=padding, bias=has_bias
        )
        torch_conv2d.type(torch.DoubleTensor)

        conv2d_layer = Conv2DLayer(in_ch, (k_size, k_size), out_ch,
                                   lambda t: torch.nn.init.normal(t, -1, 1),
                                   stride=(stride, stride),
                                   padding=(padding, padding),
                                   bias=has_bias)
        conv2d_layer.type(torch.DoubleTensor)

        input_tensor = (torch.DoubleTensor(batch_size, in_ch, height, width)
                             .uniform_(-1, 1))
        input_layer = Variable(input_tensor, requires_grad=True)
        input_torch = Variable(input_tensor.clone(), requires_grad=True)

        bias_tensor = torch.DoubleTensor(out_ch).uniform_(-1, 1)
        weights = (torch.DoubleTensor(out_ch, in_ch, k_size, k_size)
                        .uniform_(-1, 1))
        torch_conv2d.weight.data.copy_(weights)
        if has_bias:
            torch_conv2d.bias.data.copy_(bias_tensor)
        layer_weight_shape = (out_ch, in_ch * k_size * k_size)
        conv2d_layer.kernels.data.copy_(weights.view(layer_weight_shape))
        if has_bias:
            conv2d_layer.bias.data.copy_(bias_tensor.view(out_ch, 1))

        layer_result = conv2d_layer(input_layer)
        layer_result_np = layer_result.data.numpy()
        torch_result = torch_conv2d(input_torch)
        torch_result_np = torch_result.data.numpy()
        self.assertTrue(np.allclose(layer_result_np, torch_result_np))

        # verify gradient
        gradient = torch.DoubleTensor(layer_result.shape)
        layer_result.backward(gradient)
        torch_result.backward(gradient)
        self.assertTrue(
            np.allclose(input_layer.grad.data.numpy(),
                        input_torch.grad.data.numpy(),
                        equal_nan=True)
        )
        layer_weight_grad = conv2d_layer.kernels.grad
        torch_weight_grad = torch_conv2d.weight.grad.view(layer_weight_shape)
        self.assertTrue(
            np.allclose(layer_weight_grad.data.numpy(),
                        torch_weight_grad.data.numpy(),
                        equal_nan=True)
        )
        if has_bias:
            layer_bias_grad = conv2d_layer.bias.grad.view(out_ch)
            torch_bias_grad = torch_conv2d.bias.grad.view(out_ch)
            self.assertTrue(
                np.allclose(layer_bias_grad.data.numpy(),
                            torch_bias_grad.data.numpy(),
                            equal_nan=True)
            )

    def test1(self):
        self.helper_func(0)

    def test2(self):
        self.helper_func(1)

    def test3(self):
        self.helper_func(2)

    def test4(self):
        self.helper_func(3)

    def test5(self):
        self.helper_func(4)

    def test6(self):
        self.helper_func(5)

    def test7(self):
        self.helper_func(6)

    def test8(self):
        self.helper_func(7)

    def test9(self):
        self.helper_func(8)

    def test10(self):
        self.helper_func(9)


if __name__ == '__main__':
    unittest.main()

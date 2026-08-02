'''
Created on 2 August 2026

check the optional attention-gated U-Net path

@author: Dinghao Luo
'''

#%% imports
import unittest

import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight.model import build_model


#%% tests
class ModelTests(unittest.TestCase):
    def test_attention_model_keeps_the_input_shape(self):
        model = build_model({
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 4,
            'depth': 4,
            'attention_gates': True,
            })
        image = torch.rand(2, 1, 31, 35)

        output = model(image)

        self.assertEqual(output.shape, image.shape)

    def test_attention_model_backpropagates(self):
        model = build_model({
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 4,
            'depth': 2,
            'attention_gates': True,
            })
        image = torch.rand(1, 1, 16, 16)

        loss = model(image).square().mean()
        loss.backward()

        attention_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if '.attention.' in name
            ]
        self.assertTrue(attention_grads)
        self.assertTrue(all(grad is not None for grad in attention_grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in attention_grads))


if __name__ == '__main__':
    unittest.main()

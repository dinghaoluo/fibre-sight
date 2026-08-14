'''
Created on 2 August 2026

Modified on 14 August 2026

check the attention-gated U-Net experiment

@author: Dinghao Luo
'''

#%% imports
import unittest

import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight.model import build_model


#%% test
class ModelTests(unittest.TestCase):
    def test_attention_model_forward_and_backward(self):
        model = build_model({
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 4,
            'depth': 2,
            'attention_gates': True,
            })
        image = torch.rand(1, 1, 16, 16)

        output = model(image)
        output.square().mean().backward()

        self.assertEqual(output.shape, image.shape)


if __name__ == '__main__':
    unittest.main()

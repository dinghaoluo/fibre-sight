'''
Created on 25 July 2026

Modified on 14 August 2026

run an actual MPS training step when the backend is available

@author: Dinghao Luo
'''

#%% imports
import unittest

import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight.model import build_model
from fibre_sight.train_unet import segmentation_loss


#%% real backend smoke test
@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is not available')
class MPSTrainingTests(unittest.TestCase):
    def test_small_unet_forward_backward_and_step(self):
        device = torch.device('mps')
        images = torch.rand(1, 1, 32, 32, device=device)
        masks = torch.rand(1, 1, 32, 32, device=device)

        for mode in ['bce_dice', 'bce_tversky']:
            model = build_model({
                'in_channels': 1,
                'out_channels': 1,
                'base_channels': 4,
                'depth': 2,
                }).to(device)
            optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

            logits = model(images)
            loss_config = {
                'mode': mode,
                'bce_weight': 1.0,
                'seg_weight': 1.0,
                'tversky_alpha': 0.3,
                'tversky_beta': 0.7,
                }
            loss, _ = segmentation_loss(logits, masks, loss_config)
            loss.backward()
            optimiser.step()
            self.assertTrue(bool(torch.isfinite(loss).cpu()))


if __name__ == '__main__':
    unittest.main()

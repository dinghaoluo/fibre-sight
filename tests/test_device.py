'''
Created on 25 July 2026

check automatic accelerator selection and MPS training operations

@author: Dinghao Luo
'''

#%% imports
from unittest.mock import patch
import unittest

import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight._device import resolve_device
from fibre_sight.model import build_model
from fibre_sight.train_unet import segmentation_loss, use_amp, use_pinned_memory


#%% selection tests
class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with (
            patch('torch.cuda.is_available', return_value=True),
            patch('torch.backends.mps.is_available', return_value=True),
        ):
            self.assertEqual(resolve_device('auto').type, 'cuda')

    def test_auto_uses_mps_before_cpu(self):
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=True),
            patch('torch.backends.mps.is_built', return_value=True),
        ):
            self.assertEqual(resolve_device('auto').type, 'mps')

    def test_auto_falls_back_to_cpu(self):
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=False),
        ):
            self.assertEqual(resolve_device('auto').type, 'cpu')

    def test_explicit_cpu_stays_on_cpu(self):
        self.assertEqual(resolve_device('cpu').type, 'cpu')

    def test_unavailable_mps_has_a_clear_error(self):
        with (
            patch('torch.backends.mps.is_built', return_value=True),
            patch('torch.backends.mps.is_available', return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, 'not available on this Mac'):
                resolve_device('mps')

    def test_mps_keeps_cuda_specific_options_off(self):
        config = {'train': {'amp': True, 'pin_memory': True}}
        device = torch.device('mps')
        self.assertFalse(use_amp(config, device))
        self.assertFalse(use_pinned_memory(config, device))

    def test_cuda_can_use_amp_and_pinned_memory(self):
        config = {'train': {'amp': True, 'pin_memory': True}}
        device = torch.device('cuda')
        self.assertTrue(use_amp(config, device))
        self.assertTrue(use_pinned_memory(config, device))


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
            loss, _ = segmentation_loss(logits, masks, {'mode': mode})
            loss.backward()
            optimiser.step()

            self.assertTrue(bool(torch.isfinite(loss).detach().cpu()))


if __name__ == '__main__':
    unittest.main()

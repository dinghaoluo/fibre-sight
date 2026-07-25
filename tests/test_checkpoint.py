'''
Created on 24 July 2026
load the public checkpoint on CPU and check its model state

@author: Dinghao Luo
'''

#%% imports
import hashlib
import unittest

import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import get_default_checkpoint
from fibre_sight.model import build_model


#%% constants
EXPECTED_SHA256 = 'A715D1D3CA7E689A374446437F2ABBCA7113AC75F648D17D19F42759CFF1BA89'
EXPECTED_KEYS = {
    'format_version',
    'model_name',
    'training_epoch',
    'validation_pixel_dice',
    'model_state',
    'model_config',
    'data_config',
    'postprocess_config',
    }


#%% helpers
def _string_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(_string_values(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_string_values(item))
        return strings
    return []


#%% tests
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = get_default_checkpoint('ch2_v1')
        cls.checkpoint = torch.load(cls.path, map_location='cpu', weights_only=True)

    def test_checkpoint_identity_and_public_metadata(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest().upper()

        self.assertEqual(digest, EXPECTED_SHA256)
        self.assertEqual(set(self.checkpoint), EXPECTED_KEYS)
        self.assertEqual(self.checkpoint['format_version'], 1)
        self.assertEqual(self.checkpoint['model_name'], 'ch2_v1')
        self.assertEqual(self.checkpoint['training_epoch'], 37)
        self.assertAlmostEqual(self.checkpoint['validation_pixel_dice'], 0.7821986334664481)
        self.assertEqual(self.checkpoint['data_config'], {'normalise_percentiles': [1, 99.7]})
        self.assertEqual(
            self.checkpoint['postprocess_config'],
            {'threshold': 0.25, 'min_size': 45, 'max_size': None, 'tta': True},
            )

    def test_checkpoint_has_no_private_string_values(self):
        metadata = {
            key: value
            for key, value in self.checkpoint.items()
            if key != 'model_state'
            }
        self.assertEqual(_string_values(metadata), ['ch2_v1'])

    def test_model_state_loads_strictly_on_cpu(self):
        state = self.checkpoint['model_state']
        model = build_model(self.checkpoint['model_config'])
        incompatible = model.load_state_dict(state, strict=True)

        self.assertEqual(len(state), 92)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(tensor.device.type == 'cpu' for tensor in state.values()))
        self.assertTrue(all(parameter.device.type == 'cpu' for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()

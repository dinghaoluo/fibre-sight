'''
Created on 24 July 2026

run a fixed CPU prediction and connected-component smoke test

@author: Dinghao Luo
'''

#%% imports
import unittest

import numpy as np
import torch

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import get_default_checkpoint
from fibre_sight.postprocess import probability_to_roi_dict
from fibre_sight.predict_rois import load_trained_model, predict_probability


#%% fixtures
def _fixed_image():
    ypix, xpix = np.mgrid[:32, :32]
    image = 0.25 * xpix + 0.1 * ypix
    image += 18 * np.exp(-((xpix - 9) ** 2 + (ypix - 11) ** 2) / 8)
    image += 12 * np.exp(-((xpix - 23) ** 2) / 2 - ((ypix - 20) ** 2) / 18)
    return image.astype(np.float32)


#%% tests
class PredictionTests(unittest.TestCase):
    def test_fixed_cpu_prediction_and_postprocess(self):
        device = torch.device('cpu')
        model, checkpoint = load_trained_model(get_default_checkpoint('ch2_v1'), device)
        probability = predict_probability(
            _fixed_image(),
            model,
            device,
            normalise_percentiles=checkpoint['data_config']['normalise_percentiles'],
            tta=checkpoint['postprocess_config']['tta'],
            )
        roi_dict, labelled = probability_to_roi_dict(
            probability,
            threshold=checkpoint['postprocess_config']['threshold'],
            min_size=checkpoint['postprocess_config']['min_size'],
            max_size=checkpoint['postprocess_config']['max_size'],
            )

        self.assertEqual(probability.shape, (32, 32))
        self.assertEqual(probability.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertGreaterEqual(float(np.min(probability)), 0.0)
        self.assertLessEqual(float(np.max(probability)), 1.0)
        observed = np.array([
            np.min(probability),
            np.max(probability),
            np.mean(probability),
            probability[0, 0],
            probability[8, 8],
            probability[16, 16],
            probability[24, 24],
            probability[31, 31],
            ])
        expected = np.array([
            2.9229897e-05,
            9.6068010e-03,
            4.7185208e-04,
            3.9604850e-04,
            1.3721889e-04,
            1.3231132e-04,
            8.9582091e-04,
            9.6068010e-03,
            ])
        np.testing.assert_allclose(observed, expected, rtol=1e-3, atol=1e-7)
        self.assertEqual(labelled.shape, probability.shape)
        self.assertEqual(len(roi_dict), 0)
        self.assertEqual(len(roi_dict), int(np.max(labelled)))

    def test_fixed_component_filtering(self):
        probability = np.zeros((12, 14), dtype=np.float32)
        probability[1:4, 1:4] = 0.75
        probability[6:8, 2:6] = 0.6
        probability[9, 10] = 0.9

        roi_dict, labelled = probability_to_roi_dict(
            probability,
            threshold=0.5,
            min_size=4,
            max_size=8,
            )

        self.assertEqual(list(roi_dict), [1])
        self.assertEqual(int(np.sum(labelled > 0)), 8)
        np.testing.assert_array_equal(roi_dict[1]['xpix'], np.tile(np.arange(2, 6), 2))
        np.testing.assert_array_equal(roi_dict[1]['ypix'], np.repeat(np.arange(6, 8), 4))


if __name__ == '__main__':
    unittest.main()

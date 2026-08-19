'''
Created on 24 July 2026

Modified on 14 August 2026

run one bundled example through CPU prediction

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import tempfile
import unittest

import numpy as np

from support import PROJECT_ROOT, add_source_to_path

add_source_to_path()

from fibre_sight.api import BUNDLED_CHECKPOINT
from fibre_sight.predict_rois import predict_roi_dict


#%% test
class ExamplePredictionTests(unittest.TestCase):
    def test_bundled_example_prediction(self):
        image_path = PROJECT_ROOT / 'benchmarking' / 'sources' / 'lab-fibresight-demo-test.npy'
        image = np.load(image_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / 'predicted_ROI_dict.npy'
            roi_dict, labelled, probability = predict_roi_dict(
                image_path,
                BUNDLED_CHECKPOINT,
                out_path=out_path,
                device='cpu',
                )
            self.assertTrue(out_path.is_file())

        self.assertGreater(len(roi_dict), 0)
        self.assertEqual(labelled.shape, image.shape)
        self.assertEqual(probability.shape, image.shape)


if __name__ == '__main__':
    unittest.main()

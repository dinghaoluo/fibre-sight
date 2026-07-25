'''
Created on 24 July 2026
run the public examples through the bundled CPU prediction path

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from support import PROJECT_ROOT, source_environment


#%% paths
EXAMPLE_IMAGES = (
    PROJECT_ROOT / 'examples' / 'demo_test_ref_mat_ch2.npy',
    PROJECT_ROOT / 'examples' / 'demo_train_01_ref_mat_ch2.npy',
    PROJECT_ROOT / 'examples' / 'demo_train_02_ref_mat_ch2.npy',
    )
PRIMARY_EXAMPLE = EXAMPLE_IMAGES[0]


#%% helpers
def _check_roi_dict(test_case, roi_dict, shape):
    test_case.assertIsInstance(roi_dict, dict)
    test_case.assertGreater(len(roi_dict), 0)

    for roi in roi_dict.values():
        xpix = np.asarray(roi['xpix'])
        ypix = np.asarray(roi['ypix'])
        test_case.assertEqual(xpix.ndim, 1)
        test_case.assertEqual(ypix.ndim, 1)
        test_case.assertEqual(len(xpix), len(ypix))
        test_case.assertGreater(len(xpix), 0)
        test_case.assertTrue(np.all((0 <= xpix) & (xpix < shape[1])))
        test_case.assertTrue(np.all((0 <= ypix) & (ypix < shape[0])))


#%% tests
class ExamplePredictionTests(unittest.TestCase):
    def test_example_images(self):
        for image_path in EXAMPLE_IMAGES:
            with self.subTest(image=image_path.name):
                image = np.load(image_path, allow_pickle=False)
                self.assertEqual(image.shape, (256, 256))
                self.assertEqual(image.dtype, np.uint8)
                self.assertTrue(np.all(np.isfinite(image)))

    def test_public_cpu_prediction(self):
        with tempfile.TemporaryDirectory(prefix='fibre sight example ') as temp_dir:
            out_path = Path(temp_dir) / 'demo_predicted_ROI_dict.npy'
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'fibre_sight.predict_rois',
                    '--image',
                    str(PRIMARY_EXAMPLE),
                    '--out',
                    str(out_path),
                    '--device',
                    'cpu',
                    ],
                cwd=PROJECT_ROOT,
                env=source_environment(),
                capture_output=True,
                text=True,
                timeout=180,
                )

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn('predicted ROIs:', output)
            self.assertTrue(out_path.is_file())
            roi_dict = np.load(out_path, allow_pickle=True).item()

        _check_roi_dict(self, roi_dict, (256, 256))


if __name__ == '__main__':
    unittest.main()

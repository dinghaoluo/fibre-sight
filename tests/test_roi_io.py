'''
Created on 24 July 2026

Modified on 14 August 2026

check ROI dictionary validity and round trips

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import tempfile
import unittest

import numpy as np

from support import add_source_to_path

add_source_to_path()

from fibre_sight.roi_io import (
    labels_to_roi_dict,
    load_roi_dict,
    roi_dict_to_label,
    save_roi_dict,
    )


#%% tests
class ROIIOTests(unittest.TestCase):
    def test_invalid_coordinates_fail(self):
        invalid_rois = [
            {1: {'xpix': [1, 2], 'ypix': [1]}},
            {1: {'xpix': [6], 'ypix': [1]}},
            ]
        for roi_dict in invalid_rois:
            with self.subTest(roi_dict=roi_dict):
                with self.assertRaises(ValueError):
                    roi_dict_to_label(roi_dict, (6, 6))

    def test_label_and_dictionary_round_trip(self):
        labelled = np.zeros((7, 9), dtype=np.int32)
        labelled[1:3, 1:3] = 4
        labelled[4, 5:8] = 9

        roi_dict = labels_to_roi_dict(labelled)
        restored, areas = roi_dict_to_label(roi_dict, labelled.shape)

        expected = np.zeros_like(labelled)
        expected[1:3, 1:3] = 1
        expected[4, 5:8] = 2
        self.assertEqual(areas, [4, 3])
        np.testing.assert_array_equal(restored, expected)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'ROI_dict.npy'
            save_roi_dict(roi_dict, path)
            loaded = load_roi_dict(path)

        for roi_id in roi_dict:
            np.testing.assert_array_equal(loaded[roi_id]['xpix'], roi_dict[roi_id]['xpix'])
            np.testing.assert_array_equal(loaded[roi_id]['ypix'], roi_dict[roi_id]['ypix'])


if __name__ == '__main__':
    unittest.main()

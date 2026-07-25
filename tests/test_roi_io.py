'''
Created on 24 July 2026
check ROI dictionary, label image, and file round trips

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
    clean_roi_dict,
    labels_to_roi_dict,
    load_roi_dict,
    roi_dict_to_label,
    save_roi_dict,
    )


#%% tests
class ROIIOTests(unittest.TestCase):
    def test_malformed_entries_are_counted(self):
        roi_dict = {
            1: {'xpix': [1, 2], 'ypix': [1, 2]},
            2: {'xpix': ['bad'], 'ypix': [3]},
            3: {'xpix': [4]},
            }

        cleaned, invalid = clean_roi_dict(roi_dict, (6, 6))

        self.assertEqual(list(cleaned), [1])
        self.assertEqual(invalid, 2)

    def test_label_and_dictionary_round_trip(self):
        labelled = np.zeros((7, 9), dtype=np.int32)
        labelled[1:3, 1:3] = 4
        labelled[4, 5:8] = 9

        roi_dict = labels_to_roi_dict(labelled)
        restored, areas, invalid = roi_dict_to_label(roi_dict, labelled.shape)

        expected = np.zeros_like(labelled)
        expected[1:3, 1:3] = 1
        expected[4, 5:8] = 2
        self.assertEqual(list(roi_dict), [1, 2])
        self.assertEqual(areas, [4, 3])
        self.assertEqual(invalid, 0)
        np.testing.assert_array_equal(restored, expected)

    def test_dictionary_file_round_trip(self):
        labelled = np.zeros((6, 8), dtype=np.int32)
        labelled[1:3, 2:5] = 1
        labelled[4, 6:8] = 2
        roi_dict = labels_to_roi_dict(labelled)

        with tempfile.TemporaryDirectory(prefix='fibre sight roi ') as temp_dir:
            path = Path(temp_dir) / 'round trip ROI_dict.npy'
            save_roi_dict(roi_dict, path)
            loaded = load_roi_dict(path)

        self.assertEqual(list(loaded), list(roi_dict))
        for roi_id in roi_dict:
            np.testing.assert_array_equal(loaded[roi_id]['xpix'], roi_dict[roi_id]['xpix'])
            np.testing.assert_array_equal(loaded[roi_id]['ypix'], roi_dict[roi_id]['ypix'])


if __name__ == '__main__':
    unittest.main()

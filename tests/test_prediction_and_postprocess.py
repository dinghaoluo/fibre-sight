'''
Created on 24 July 2026

Modified on 14 August 2026

check connected-component filtering used by ROI prediction

@author: Dinghao Luo
'''

#%% imports
import unittest

import numpy as np

from support import add_source_to_path

add_source_to_path()

from fibre_sight.postprocess import probability_to_roi_dict


#%% test
class PostprocessTests(unittest.TestCase):
    def test_component_filtering(self):
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


if __name__ == '__main__':
    unittest.main()

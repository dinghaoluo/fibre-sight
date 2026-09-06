'''
Created on 1 August 2026

Modified on 14 August 2026

check that augmentation keeps the image and ROI mask aligned

@author: Dinghao Luo
'''

#%% imports
import unittest

import numpy as np

from support import add_source_to_path

add_source_to_path()

from fibre_sight.image_ops import augment_pair


#%% fixed random choices
class FixedRNG:
    def __init__(self, rotation=0):
        self.rotation = rotation

    def random(self):
        return 1.0

    def integers(self, low, high):
        return self.rotation

    def normal(self, mean, sd, size):
        return np.full(size, mean + sd)


#%% tests
class AugmentationTests(unittest.TestCase):
    def test_rotation_and_noise_keep_the_mask_aligned(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
        mask = image >= 0.4

        augmented_image, augmented_mask = augment_pair(
            image,
            mask,
            FixedRNG(rotation=1),
            rotation_90=True,
            intensity_jitter=0,
            noise_sd=0.1,
            )

        np.testing.assert_allclose(augmented_image, np.rot90(image, 1) + 0.1)
        np.testing.assert_array_equal(augmented_mask, np.rot90(mask, 1))


if __name__ == '__main__':
    unittest.main()

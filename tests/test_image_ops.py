'''
Created on 1 August 2026

check paired rotation and image-only noise augmentation

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
    def test_rotation_keeps_image_and_mask_aligned(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
        mask = image >= 0.4

        augmented_image, augmented_mask = augment_pair(
            image,
            mask,
            FixedRNG(rotation=1),
            rotation_90=True,
            intensity_jitter=0,
            noise_sd=0,
            )

        np.testing.assert_array_equal(augmented_image, np.rot90(image, 1))
        np.testing.assert_array_equal(augmented_mask, np.rot90(mask, 1))

    def test_noise_changes_the_image_only(self):
        image = np.full((3, 3), 0.5, dtype=np.float32)
        mask = np.eye(3, dtype=np.float32)

        augmented_image, augmented_mask = augment_pair(
            image,
            mask,
            FixedRNG(),
            rotation_90=False,
            intensity_jitter=0,
            noise_sd=0.1,
            )

        np.testing.assert_allclose(augmented_image, 0.6)
        np.testing.assert_array_equal(augmented_mask, mask)


if __name__ == '__main__':
    unittest.main()

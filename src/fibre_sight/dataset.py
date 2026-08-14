'''
Created on 8 April 2026

Modified on 23 July 2026 to repair the Dataset import and keep target construction here
Modified on 1 August 2026 to pass augmentation choices from the training recipe
Modified on 14 August 2026

sample channel-2 image patches and their curated ROI masks

@author: Dinghao Luo
'''

#%% imports
import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import Dataset

from .image_ops import augment_pair, normalise, sample_crop_bounds
from .manifest import read_manifest
from .roi_io import load_roi_dict, roi_dict_to_mask


#%% targets
def make_target(mask, mode='foreground', support_radius=3):
    mask = np.asarray(mask).astype(bool)

    if mode == 'foreground':
        return mask[None, ...].astype(np.float32)

    if mode == 'foreground_support':
        # the wider channel was my recall-heavy trial; saved ROIs still use channel 0
        support = ndi.binary_dilation(mask, iterations=support_radius)
        target = np.stack([mask, support], axis=0)
        return target.astype(np.float32)

    raise ValueError(f'unknown target mode: {mode}')


#%% dataset
class ROIDataset(Dataset):
    def __init__(
            self,
            manifest_path,
            split='train',
            patch_size=256,
            patches_per_image=24,
            foreground_fraction=0.75,
            normalise_percentiles=(1, 99.7),
            augment=True,
            rotation_90=True,
            noise_sd=0.02,
            cache_images=False,
            target_mode='foreground',
            support_radius=3,
            seed=7,
            ):
        self.sessions = read_manifest(manifest_path, included_only=True, split=split)
        if not self.sessions:
            raise ValueError(f'no sessions found for split {split}')

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.foreground_fraction = foreground_fraction
        self.normalise_percentiles = normalise_percentiles
        self.augment = augment
        self.rotation_90 = rotation_90
        self.noise_sd = noise_sd
        self.cache_images = cache_images
        self.target_mode = target_mode
        self.support_radius = support_radius
        self.seed = seed
        self.epoch = 0
        self.cache = {}

    def __len__(self):
        return len(self.sessions) * self.patches_per_image

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        session = self.sessions[idx % len(self.sessions)]
        image, mask = self.load_pair(session)

        # the epoch changes the crops whilst the seed keeps a rerun reproducible
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + idx)
        top, left, height, width = sample_crop_bounds(
            mask,
            self.patch_size,
            rng,
            foreground_fraction=self.foreground_fraction,
            )
        image = image[top:top + height, left:left + width]
        mask = mask[top:top + height, left:left + width]

        if self.augment:
            image, mask = augment_pair(
                image,
                mask,
                rng,
                rotation_90=self.rotation_90,
                noise_sd=self.noise_sd,
                )

        target = make_target(
            mask,
            mode=self.target_mode,
            support_radius=self.support_radius,
            )
        return {
            'image': torch.from_numpy(image[None, ...]),
            'mask': torch.from_numpy(target),
            }

    def load_pair(self, session):
        cache_key = session['session']
        if self.cache_images and cache_key in self.cache:
            return self.cache[cache_key]

        image = np.load(session['image_path'])
        roi_dict = load_roi_dict(session['roi_path'])
        mask = roi_dict_to_mask(roi_dict, image.shape).astype(np.float32)
        image = normalise(
            image,
            low=self.normalise_percentiles[0],
            high=self.normalise_percentiles[1],
            )

        if self.cache_images:
            self.cache[cache_key] = (image, mask)
        return image, mask

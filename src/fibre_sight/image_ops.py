'''
Created on 5 April 2026

Modified on 1 August 2026 to expose the lossless rotation decision
Modified on 14 August 2026

normalisation, crop sampling, and augmentation for channel-2 images

@author: Dinghao Luo
'''

#%% imports
import numpy as np


#%% normalisation
def normalise(image, low=1, high=99.7):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image)

    # per-image percentiles keep acquisition brightness out of the labels
    lo, hi = np.percentile(image[finite], [low, high])
    if hi <= lo:
        return np.zeros_like(image)

    image = (image - lo) / (hi - lo)
    image[~finite] = 0
    return np.clip(image, 0, 1).astype(np.float32)


#%% crops
def sample_crop_bounds(mask, patch_size, rng, foreground_fraction=0.75):
    height, width = mask.shape
    if patch_size > height or patch_size > width:
        raise ValueError('patch size exceeds the image dimensions')

    # keep some background-only patches instead of teaching the model that
    # every crop contains an axon
    use_foreground = rng.random() < foreground_fraction and np.any(mask)
    if use_foreground:
        ypix, xpix = np.where(mask)
        idx = rng.integers(0, len(ypix))
        top = int(ypix[idx]) - patch_size // 2
        left = int(xpix[idx]) - patch_size // 2
    else:
        top = rng.integers(0, height - patch_size + 1)
        left = rng.integers(0, width - patch_size + 1)

    top = int(np.clip(top, 0, height - patch_size))
    left = int(np.clip(left, 0, width - patch_size))
    return top, left, patch_size, patch_size


#%% augmentation
def augment_pair(
        image,
        mask,
        rng,
        rotation_90=True,
        intensity_jitter=0.15,
        noise_sd=0.02,
        ):
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    # 1 August 2026: keep rotations lossless; thin masks do not need interpolation
    if rotation_90:
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

    if intensity_jitter > 0:
        scale = 1 + rng.uniform(-intensity_jitter, intensity_jitter)
        offset = rng.uniform(-intensity_jitter, intensity_jitter) * 0.25
        image = image * scale + offset

    if noise_sd > 0:
        image = image + rng.normal(0, noise_sd, size=image.shape)

    image = np.clip(image, 0, 1).astype(np.float32)
    mask = mask.astype(np.float32)
    return image, mask

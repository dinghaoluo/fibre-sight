'''
Created on 21 April 2026

Modified on 14 August 2026

turn probability maps into one ROI dictionary entry per connected component

@author: Dinghao Luo
'''

#%% imports
import numpy as np
from scipy import ndimage as ndi

from .roi_io import labels_to_roi_dict


#%% labels
def probability_to_labels(probability, threshold=0.5, min_size=30, max_size=None):
    mask = np.asarray(probability) >= threshold
    labelled, _ = ndi.label(mask)
    labelled = _filter_labels(labelled, min_size=min_size, max_size=max_size)
    return labelled


def _filter_labels(labelled, min_size=30, max_size=None):
    labelled = np.asarray(labelled)
    out = np.zeros_like(labelled, dtype=np.int32)
    next_id = 1

    for label_id in sorted(np.unique(labelled)):
        if label_id == 0:
            continue

        mask = labelled == label_id
        area = int(np.sum(mask))
        if area < min_size:
            continue
        if max_size is not None and area > max_size:
            continue

        out[mask] = next_id
        next_id += 1

    return out


def probability_to_roi_dict(probability, threshold=0.5, min_size=30, max_size=None):
    labelled = probability_to_labels(
        probability,
        threshold=threshold,
        min_size=min_size,
        max_size=max_size,
        )
    return labels_to_roi_dict(labelled), labelled

'''
Created on 5 April 2026

Modified on 24 July 2026 to build ROI masks without the lab utility module
Modified on 14 August 2026

convert between xpix/ypix ROI dictionaries and labelled images

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import numpy as np


#%% loading
def load_roi_dict(path):
    roi_dict = np.load(path, allow_pickle=True).item()
    if not isinstance(roi_dict, dict):
        raise ValueError(f'not an ROI dict: {path}')
    return roi_dict


def save_roi_dict(roi_dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, roi_dict)


#%% conversion
def roi_coordinates(roi_id, roi, shape):
    xpix = np.asarray(roi['xpix'], dtype=np.int64).ravel()
    ypix = np.asarray(roi['ypix'], dtype=np.int64).ravel()
    if len(xpix) != len(ypix) or len(xpix) == 0:
        raise ValueError(f'ROI {roi_id} has invalid coordinate lengths')
    if (
            np.any(xpix < 0) or np.any(xpix >= shape[1]) or
            np.any(ypix < 0) or np.any(ypix >= shape[0])
            ):
        raise ValueError(f'ROI {roi_id} has coordinates outside the image')
    return xpix, ypix


def roi_dict_to_mask(roi_dict, shape):
    labelled, _ = roi_dict_to_label(roi_dict, shape)
    return labelled > 0


def roi_dict_to_label(roi_dict, shape):
    labelled = np.zeros(shape, dtype=np.int32)
    roi_areas = []

    for new_id, (roi_id, roi) in enumerate(roi_dict.items(), start=1):
        xpix, ypix = roi_coordinates(roi_id, roi, shape)
        labelled[ypix, xpix] = new_id
        roi_areas.append(len(xpix))

    return labelled, roi_areas


def labels_to_roi_dict(labelled):
    roi_dict = {}
    next_id = 1
    for label_id in sorted(np.unique(labelled)):
        if label_id == 0:
            continue
        ypix, xpix = np.where(labelled == label_id)
        roi_dict[next_id] = {
            'xpix': xpix.astype(np.int64),
            'ypix': ypix.astype(np.int64),
            }
        next_id += 1
    return roi_dict


def roi_summary(roi_dict, shape):
    labelled, roi_areas = roi_dict_to_label(roi_dict, shape)
    mask = labelled > 0
    return {
        'roi_count': len(roi_areas),
        'positive_pixels': int(np.sum(mask)),
        'positive_fraction': float(np.mean(mask)),
        'median_roi_area_px': float(np.median(roi_areas)) if roi_areas else 0.0,
        'min_roi_area_px': int(np.min(roi_areas)) if roi_areas else 0,
        'max_roi_area_px': int(np.max(roi_areas)) if roi_areas else 0,
        }

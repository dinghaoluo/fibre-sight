'''
Created on 6 April 2026

Modified on 29 June 2026
Modified on 14 August 2026

scan labelled sessions and keep the train, validation, and test split

@author: Dinghao Luo
'''

#%% imports
from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from .roi_io import load_roi_dict, roi_summary


#%% columns
MANIFEST_COLUMNS = [
    'session',
    'animal',
    'processed_path',
    'image_path',
    'roi_path',
    'height',
    'width',
    'roi_count',
    'positive_pixels',
    'positive_fraction',
    'median_roi_area_px',
    'min_roi_area_px',
    'max_roi_area_px',
    'included',
    'exclusion_reason',
    'split',
    ]


#%% scanning
def _find_ref_paths(processed_path):
    # newer sessions use the canonical name; older exports kept the recording prefix
    canonical_paths = sorted(processed_path.glob('ref_mat_ch2.npy'))
    archive_paths = sorted(processed_path.glob('*_ref_mat_ch2.npy'))
    return canonical_paths + [
        path for path in archive_paths
        if path not in canonical_paths
        ]


def _find_ref_roi_pair(processed_path):
    ref_paths = _find_ref_paths(processed_path)
    roi_paths = sorted(processed_path.glob('*_ROI_dict.npy'))

    canonical_path = processed_path / 'ref_mat_ch2.npy'
    if canonical_path in ref_paths:
        ref_path = canonical_path
    elif len(ref_paths) == 1:
        ref_path = ref_paths[0]
    else:
        raise ValueError(f'found several channel-2 references in {processed_path}')
    if ref_path.name == 'ref_mat_ch2.npy':
        prefix = processed_path.parent.name
    else:
        prefix = ref_path.name.replace('_ref_mat_ch2.npy', '')
    if len(roi_paths) == 1:
        roi_path = roi_paths[0]
    else:
        matched_rois = [path for path in roi_paths if path.name.startswith(prefix)]
        if len(matched_rois) != 1:
            raise ValueError(f'could not match one ROI dict to {ref_path}')
        roi_path = matched_rois[0]
    return ref_path, roi_path


def _exclusion_reason(session_path):
    processed_path = session_path / 'processed_data'
    if not processed_path.exists():
        return 'no processed_data'

    if not _find_ref_paths(processed_path):
        return 'no channel-2 reference'
    if not list(processed_path.glob('*_ROI_dict.npy')):
        return 'no ROI dict'
    return ''


def scan_session(session_path):
    session_path = Path(session_path)
    processed_path = session_path / 'processed_data'
    exclusion_reason = _exclusion_reason(session_path)

    session = {
        'session': session_path.name,
        'animal': session_path.name.split('-')[0],
        'processed_path': str(processed_path),
        'image_path': '',
        'roi_path': '',
        'height': '',
        'width': '',
        'roi_count': 0,
        'positive_pixels': 0,
        'positive_fraction': 0,
        'median_roi_area_px': 0,
        'min_roi_area_px': 0,
        'max_roi_area_px': 0,
        'included': exclusion_reason == '',
        'exclusion_reason': exclusion_reason,
        'split': '',
    }

    if exclusion_reason:
        return session

    image_path, roi_path = _find_ref_roi_pair(processed_path)
    image = np.load(image_path, mmap_mode='r')
    roi_dict = load_roi_dict(roi_path)
    session.update(roi_summary(roi_dict, image.shape))
    session.update({
        'image_path': str(image_path),
        'roi_path': str(roi_path),
        'height': int(image.shape[0]),
        'width': int(image.shape[1]),
        })
    return session


def scan_source_root(source_root):
    source_root = Path(source_root)
    return [
        scan_session(session_path)
        for session_path in sorted(path for path in source_root.iterdir() if path.is_dir())
        ]


#%% splitting
def assign_session_splits(sessions, val_fraction=0.15, test_fraction=0.15, seed=42):
    # whole sessions stay together so one recording cannot enter two splits
    rng = np.random.default_rng(seed)
    included = [session for session in sessions if _as_bool(session['included'])]

    # make each animal contribute to the held-out splits when it has enough sessions
    grouped = defaultdict(list)
    for session in included:
        grouped[session['animal']].append(session)

    for animal_sessions in grouped.values():
        order = np.arange(len(animal_sessions))
        rng.shuffle(order)

        n_sessions = len(animal_sessions)
        n_test = _split_count(n_sessions, test_fraction)
        n_val = _split_count(n_sessions - n_test, val_fraction)

        for rank, session_idx in enumerate(order):
            session = animal_sessions[int(session_idx)]
            if rank < n_test:
                session['split'] = 'test'
            elif rank < n_test + n_val:
                session['split'] = 'val'
            else:
                session['split'] = 'train'

    for session in sessions:
        if not _as_bool(session['included']):
            session['split'] = ''
    return sessions


def _split_count(n_sessions, fraction):
    if fraction <= 0 or n_sessions <= 1:
        return 0
    return max(1, int(round(n_sessions * fraction)))


#%% io
def read_manifest(path, included_only=False, split=None):
    sessions = []
    with open(path, 'r', newline='') as f:
        for session in csv.DictReader(f):
            session = _coerce_session(session)
            if included_only and not session['included']:
                continue
            if split is not None and session['split'] != split:
                continue
            sessions.append(session)
    return sessions


def write_manifest(sessions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for session in sessions:
            writer.writerow({column: session[column] for column in MANIFEST_COLUMNS})


def _coerce_session(session):
    session = dict(session)
    for column in [
            'height', 'width', 'roi_count', 'positive_pixels',
            'min_roi_area_px', 'max_roi_area_px',
            ]:
        session[column] = int(session[column]) if session[column] else 0
    for column in ['positive_fraction', 'median_roi_area_px']:
        session[column] = float(session[column]) if session[column] else 0.0
    session['included'] = _as_bool(session['included'])
    return session


def _as_bool(value):
    return str(value) == 'True'

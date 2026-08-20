'''
Created on 19 August 2026

store immutable segmentation and curation runs in NWB

@author: Dinghao Luo
'''

#%% imports
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

try:
    import resource
except ImportError:  # unavailable on Windows
    resource = None

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import h5py
import numpy as np
from pynwb import NWBHDF5IO, validate
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import ImageSegmentation, OpticalChannel

from .roi_io import roi_coordinates


#%% schema
CONTROL_REFERENCE_PATH = (
    'processing/preprocessing/registration_references/control_reference'
    )
_TEXT_COLUMNS = {
    'run_name',
    'run_type',
    'source_run',
    'reference_path',
    'checkpoint_path',
    'checkpoint_sha256',
    'device',
    'created_at',
    }


def _read_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _segmentation_runs(nwbfile):
    if 'segmentation' not in nwbfile.processing:
        return {}

    table = nwbfile.processing['segmentation']['segmentation_runs']
    runs = {}
    for row_index in range(len(table)):
        run = {}
        for column_name in table.colnames:
            value = table[column_name][row_index]
            run[column_name] = (
                _read_text(value) if column_name in _TEXT_COLUMNS else value
                )
        runs[run['run_name']] = run
    return runs


def list_roi_runs(nwb_path, run_type=None):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        runs = list(_segmentation_runs(io.read()).values())
    if run_type is None:
        return runs
    return [run for run in runs if run['run_type'] == run_type]


def _check_new_run(nwbfile, run_name):
    if not run_name or '/' in run_name:
        raise ValueError('run_name must be non-empty and cannot contain /')
    if run_name in _segmentation_runs(nwbfile):
        raise ValueError(f'ROI run already exists: {run_name}')


def _read_control_reference(nwbfile):
    reference = nwbfile.processing['preprocessing'][
        'registration_references']['control_reference']
    return np.asarray(reference.data).T.copy()


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(1024 ** 2):
            digest.update(block)
    return digest.hexdigest()


def _segmentation_module(nwbfile):
    if 'segmentation' in nwbfile.processing:
        return nwbfile.processing['segmentation']

    module = nwbfile.create_processing_module(
        name='segmentation',
        description='immutable named FibreSight ROI runs',
        )
    module.add(ImageSegmentation())
    module.add(Images(
        name='probability_maps',
        description='float32 proposal probability maps in NWB x, y order',
        ))
    runs = DynamicTable(
        name='segmentation_runs',
        description='one provenance row for each immutable ROI run',
        )
    columns = {
        'run_name': 'unique immutable run name',
        'run_type': 'proposed or curated',
        'source_run': 'proposal run used for curation; empty for proposals',
        'reference_path': 'canonical registered control reference',
        'checkpoint_path': 'checkpoint used for prediction; empty for curation',
        'checkpoint_sha256': 'checkpoint SHA-256; empty for curation',
        'threshold': 'proposal probability threshold; NaN for curation',
        'min_size': 'minimum proposal area in pixels; -1 for curation',
        'max_size': 'maximum proposal area in pixels; -1 when absent or curated',
        'tta': 'whether flip test-time augmentation was used',
        'device': 'prediction device; empty for curation',
        'created_at': 'UTC creation time',
        }
    for name, description in columns.items():
        runs.add_column(name=name, description=description)
    module.add(runs)
    return module


def _control_reference_plane(nwbfile):
    plane_name = 'registered_control_reference'
    if plane_name in nwbfile.imaging_planes:
        return nwbfile.imaging_planes[plane_name]

    metadata = nwbfile.processing['preprocessing']['recording_metadata']
    control_label = _read_text(metadata['control_label'][0]) or 'unknown'
    pixel_size_um = float(metadata['pixel_size_um'][0])
    device = nwbfile.create_device(
        name='fibre_sight_preprocessed_device',
        description='original device metadata was not supplied during preprocessing',
        )
    optical_channel = OpticalChannel(
        name='control_channel',
        description='canonical registered structural control channel',
        emission_lambda=np.nan,
        )
    plane_spacing = {}
    if np.isfinite(pixel_size_um):
        plane_spacing['grid_spacing'] = [pixel_size_um * 1e-6] * 2
        plane_spacing['grid_spacing_unit'] = 'meters'
    return nwbfile.create_imaging_plane(
        name=plane_name,
        optical_channel=optical_channel,
        description='image plane of the canonical registered control reference',
        device=device,
        excitation_lambda=np.nan,
        indicator=control_label,
        location='unknown',
        **plane_spacing,
        )


def _add_roi_run(nwbfile, run_name, roi_dict, run_metadata, probability=None):
    module = _segmentation_module(nwbfile)
    plane_segmentation = module['ImageSegmentation'].create_plane_segmentation(
        name=run_name,
        description=(
            f'{run_metadata["run_type"]} ROIs on the canonical control reference'
            ),
        imaging_plane=_control_reference_plane(nwbfile),
        )
    reference_shape = _read_control_reference(nwbfile).shape
    for roi_id, roi in roi_dict.items():
        xpix, ypix = roi_coordinates(roi_id, roi, reference_shape)
        plane_segmentation.add_roi(
            id=int(roi_id),
            pixel_mask=list(zip(xpix, ypix, np.ones(len(xpix), dtype=float))),
            )
    if not roi_dict:
        pixel_mask_dtype = np.dtype([
            ('x', np.uint32),
            ('y', np.uint32),
            ('weight', np.float32),
            ])
        plane_segmentation.add_column(
            name='pixel_mask',
            description='pixel masks for each ROI',
            data=np.empty(0, dtype=pixel_mask_dtype),
            index=True,
            )

    if probability is not None:
        module['probability_maps'].add_image(GrayscaleImage(
            name=run_name,
            data=H5DataIO(
                np.asarray(probability, dtype=np.float32).T,
                compression='gzip',
                compression_opts=1,
                shuffle=True,
                fletcher32=True,
                ),
            description='proposal probability on the canonical control reference',
            ))
    module['segmentation_runs'].add_row(**run_metadata)


def _read_plane_segmentation(plane_segmentation):
    if 'pixel_mask' not in plane_segmentation.colnames:
        return {}

    roi_dict = {}
    for row_index, roi_id in enumerate(np.asarray(plane_segmentation.id)):
        pixels = np.asarray(plane_segmentation['pixel_mask'][row_index])
        roi_dict[int(roi_id)] = {
            'xpix': np.asarray(pixels['x'], dtype=np.int64),
            'ypix': np.asarray(pixels['y'], dtype=np.int64),
            }
    return roi_dict


#%% readers
def load_roi_run(nwb_path, run_name):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        nwbfile = io.read()
        runs = _segmentation_runs(nwbfile)
        if run_name not in runs:
            raise KeyError(f'ROI run does not exist: {run_name}')

        module = nwbfile.processing['segmentation']
        probability = None
        if run_name in module['probability_maps'].images:
            probability = np.asarray(
                module['probability_maps'][run_name].data,
                dtype=np.float32,
                ).T.copy()
        return {
            'run_name': run_name,
            'roi_dict': _read_plane_segmentation(
                module['ImageSegmentation'][run_name]
                ),
            'probability': probability,
            'reference': _read_control_reference(nwbfile),
            'provenance': runs[run_name],
            }


#%% transactional writes
def _segmentation_partial_path(nwb_path):
    return nwb_path.with_name(f'{nwb_path.stem}.segmenting.partial.nwb')


def _peak_memory_bytes():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def _registered_movie_state(path):
    state = {}
    with h5py.File(path, 'r') as file:
        for channel in ('signal', 'control'):
            data = file[f'processing/preprocessing/registered_{channel}/data']
            frame_indices = sorted({0, len(data) // 2, len(data) - 1})
            state[channel] = {
                'shape': data.shape,
                'dtype': data.dtype.str,
                'chunks': data.chunks,
                'compression': data.compression,
                'compression_opts': data.compression_opts,
                'shuffle': data.shuffle,
                'fletcher32': data.fletcher32,
                'scaleoffset': data.scaleoffset,
                'frames': {
                    frame_index: hashlib.sha256(
                        np.asarray(data[frame_index]).tobytes()
                        ).hexdigest()
                    for frame_index in frame_indices
                    },
                }
    return state


def _verify_roi_run(
        path,
        run_name,
        roi_dict,
        probability,
        registered_movie_state,
        ):
    loaded = load_roi_run(path, run_name)
    if list(loaded['roi_dict']) != [int(roi_id) for roi_id in roi_dict]:
        raise AssertionError('stored ROI identifiers changed')
    for roi_id, roi in roi_dict.items():
        stored_roi = loaded['roi_dict'][int(roi_id)]
        np.testing.assert_array_equal(
            stored_roi['xpix'], np.asarray(roi['xpix']).ravel()
            )
        np.testing.assert_array_equal(
            stored_roi['ypix'], np.asarray(roi['ypix']).ravel()
            )
    if probability is None:
        if loaded['probability'] is not None:
            raise AssertionError('curated run unexpectedly contains a probability map')
    elif not np.array_equal(
            loaded['probability'],
            np.asarray(probability, dtype=np.float32),
            ):
        raise AssertionError('stored probability map changed')
    if _registered_movie_state(path) != registered_movie_state:
        raise AssertionError('registered movie storage or seeded frames changed')


def _append_roi_run_transactionally(
        nwb_path,
        run_name,
        roi_dict,
        run_metadata,
        probability=None,
        ):
    nwb_path = Path(nwb_path)
    partial_path = _segmentation_partial_path(nwb_path)
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')

    original_size = nwb_path.stat().st_size
    registered_movie_state = _registered_movie_state(nwb_path)
    copy_start = perf_counter()
    shutil.copyfile(nwb_path, partial_path)
    copy_time_s = perf_counter() - copy_start

    with NWBHDF5IO(partial_path, 'a') as io:
        nwbfile = io.read()
        _check_new_run(nwbfile, run_name)
        _add_roi_run(nwbfile, run_name, roi_dict, run_metadata, probability)
        io.write(nwbfile)

    validation_errors = [str(error) for error in validate(path=partial_path)]
    if validation_errors:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    _verify_roi_run(
        partial_path,
        run_name,
        roi_dict,
        probability,
        registered_movie_state,
        )
    with partial_path.open('rb') as file:
        os.fsync(file.fileno())
    file_size_bytes = partial_path.stat().st_size
    os.replace(partial_path, nwb_path)
    return {
        'output_path': nwb_path,
        'run_name': run_name,
        'roi_count': len(roi_dict),
        'copy_time_s': copy_time_s,
        'file_size_bytes': file_size_bytes,
        'file_size_increase_bytes': file_size_bytes - original_size,
        'validation_errors': validation_errors,
        'peak_memory_bytes': _peak_memory_bytes(),
        }


def save_curated_rois(nwb_path, run_name, roi_dict, source_run):
    nwb_path = Path(nwb_path)
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        _check_new_run(nwbfile, run_name)
        runs = _segmentation_runs(nwbfile)
        if source_run not in runs or runs[source_run]['run_type'] != 'proposed':
            raise ValueError(f'source proposal does not exist: {source_run}')
        reference_shape = _read_control_reference(nwbfile).shape
    for roi_id, roi in roi_dict.items():
        roi_coordinates(roi_id, roi, reference_shape)

    run_metadata = {
        'run_name': run_name,
        'run_type': 'curated',
        'source_run': source_run,
        'reference_path': CONTROL_REFERENCE_PATH,
        'checkpoint_path': '',
        'checkpoint_sha256': '',
        'threshold': np.nan,
        'min_size': -1,
        'max_size': -1,
        'tta': False,
        'device': '',
        'created_at': datetime.now(timezone.utc).isoformat(),
        }
    return _append_roi_run_transactionally(
        nwb_path,
        run_name,
        roi_dict,
        run_metadata,
        )

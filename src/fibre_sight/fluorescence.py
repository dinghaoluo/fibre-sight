'''
Created on 19 August 2026
Modified on 22 August 2026 to use piecewise per-pixel validity

extract ROI and surrounding fluorescence from registered NWB movies

@author: Dinghao Luo
'''

#%% imports
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import perf_counter

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import numpy as np
from pynwb import NWBHDF5IO, validate
from pynwb.ophys import Fluorescence, RoiResponseSeries
from scipy import ndimage as ndi

from .nwb_segmentation import (
    _peak_memory_bytes,
    _registered_movie_state,
    _segmentation_runs,
    load_roi_run,
    )
from .preprocessing import registration_valid_mask


#%% defaults
SURROUND_METHOD = 'adaptive'
SURROUND_INNER_PX = 5
SURROUND_OUTER_PX = 8
SURROUND_MIN_PIXELS = 350
SURROUND_EXPANSION_PX = 5
FRAME_BATCH_SIZE = 32
STATISTICS = ('mean', 'median', 'iqr', 'valid_fraction')


#%% masks and measurements
def _roi_and_surround_coordinates(
        roi_dict,
        image_shape,
        surround_method,
        surround_inner_px,
        surround_outer_px,
        surround_min_pixels,
        ):
    occupied_roi_pixels = np.zeros(image_shape, dtype=bool)
    roi_masks = {}
    for roi_id, roi in roi_dict.items():
        roi_mask = np.zeros(image_shape, dtype=bool)
        roi_mask[roi['ypix'], roi['xpix']] = True
        roi_masks[roi_id] = roi_mask
        occupied_roi_pixels |= roi_mask

    area_coordinates = {'roi': [], 'surround': []}
    four_connected = ndi.generate_binary_structure(2, 1)
    for roi_id in roi_dict:
        roi_mask = roi_masks[roi_id]
        roi_ypix, roi_xpix = np.nonzero(roi_mask)
        if surround_method == 'fixed':
            distance_from_roi = ndi.distance_transform_edt(~roi_mask)
            surround_mask = (
                (distance_from_roi > surround_inner_px)
                & (distance_from_roi <= surround_outer_px)
                & ~occupied_roi_pixels
                )
        else:
            inner_mask = roi_mask.copy()
            if surround_inner_px:
                inner_mask = ndi.binary_dilation(
                    inner_mask,
                    structure=four_connected,
                    iterations=surround_inner_px,
                    )
            expanded_mask = inner_mask
            surround_mask = np.zeros(image_shape, dtype=bool)
            while np.count_nonzero(surround_mask) <= surround_min_pixels:
                next_mask = ndi.binary_dilation(
                    expanded_mask,
                    structure=four_connected,
                    iterations=SURROUND_EXPANSION_PX,
                    )
                if np.array_equal(next_mask, expanded_mask):
                    break
                expanded_mask = next_mask
                surround_mask = (
                    expanded_mask & ~inner_mask & ~occupied_roi_pixels
                    )

        surround_ypix, surround_xpix = np.nonzero(surround_mask)
        area_coordinates['roi'].append((roi_ypix, roi_xpix))
        area_coordinates['surround'].append((surround_ypix, surround_xpix))
    return area_coordinates


def _measure_pixels(frames, valid_bounds, ypix, xpix, valid_masks=None):
    measurements = np.full((len(frames), len(STATISTICS)), np.nan, dtype=np.float32)
    if len(ypix) == 0:
        return measurements

    if valid_masks is None:
        valid_pixels = (
            (ypix[None, :] >= valid_bounds[:, 0, None])
            & (ypix[None, :] < valid_bounds[:, 1, None])
            & (xpix[None, :] >= valid_bounds[:, 2, None])
            & (xpix[None, :] < valid_bounds[:, 3, None])
            )
    else:
        valid_pixels = valid_masks[:, ypix, xpix]
    valid_pixel_count = np.sum(valid_pixels, axis=1)
    measurements[:, 3] = valid_pixel_count / len(ypix)
    frames_with_valid_pixels = valid_pixel_count > 0
    if not np.any(frames_with_valid_pixels):
        return measurements

    pixel_values = frames[:, ypix, xpix].astype(np.float32)
    pixel_values = pixel_values[frames_with_valid_pixels]
    pixel_values[~valid_pixels[frames_with_valid_pixels]] = np.nan
    measurements[frames_with_valid_pixels, 0] = (
        np.nansum(pixel_values, axis=1)
        / valid_pixel_count[frames_with_valid_pixels]
        )
    measurements[frames_with_valid_pixels, 1] = np.nanmedian(
        pixel_values, axis=1)
    quartiles = np.nanpercentile(pixel_values, (25, 75), axis=1)
    measurements[frames_with_valid_pixels, 2] = quartiles[1] - quartiles[0]
    return measurements


def _read_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _registration_valid_masks(preprocessing, frame_indices, channel, shape):
    bounds = preprocessing['registered_valid_bounds']
    piecewise_table = (
        preprocessing['piecewise_registration']
        if 'piecewise_registration' in preprocessing.data_interfaces else None
        )
    registration_table = (
        preprocessing['registration_model']
        if 'registration_model' in preprocessing.data_interfaces else None
        )
    registration_channel = 'control'
    if (
            registration_table is not None
            and 'registration_channel' in registration_table.colnames
            ):
        registration_channel = _read_text(
            registration_table['registration_channel'][0])
    signal_to_control_offset = (0, 0)
    if 'channel_alignment' in preprocessing.data_interfaces:
        alignment = preprocessing['channel_alignment']
        signal_to_control_offset = (
            alignment['dy_px'][0],
            alignment['dx_px'][0],
            )
    rigid_translation = (
        preprocessing['rigid_translation'].data
        if 'rigid_translation' in preprocessing.data_interfaces else None
        )
    coefficients = (
        preprocessing['piecewise_spline_coefficients'].data
        if 'piecewise_spline_coefficients' in preprocessing.data_interfaces else None
        )
    grid = (
        preprocessing['piecewise_spline_grid']
        if 'piecewise_spline_grid' in preprocessing.data_interfaces else None
        )
    control_y = (
        np.unique(np.asarray(grid['y_px'])) if grid is not None else None)
    control_x = (
        np.unique(np.asarray(grid['x_px'])) if grid is not None else None)
    if control_y is not None and len(control_y) > 1:
        inferred_tile_size = np.diff(control_y)[0]
    else:
        inferred_tile_size = None

    models_used = [
        'rigid' if piecewise_table is None else _read_text(
            piecewise_table['model_used'][frame_i])
        for frame_i in frame_indices
        ]
    if 'piecewise_rigid' not in models_used:
        return None

    masks = np.empty((len(frame_indices), *shape), dtype=bool)
    for mask_i, frame_i in enumerate(frame_indices):
        valid_bounds = [
            bounds[f'{channel}_{coordinate}'][frame_i]
            for coordinate in ('y0', 'y1', 'x0', 'x1')
            ]
        model_used = models_used[mask_i]
        kwargs = {}
        if model_used == 'piecewise_rigid':
            if coefficients is None or grid is None:
                raise ValueError(
                    'piecewise registration is missing spline validity metadata')
            tile_size = (
                piecewise_table['tile_size_px'][frame_i]
                if 'tile_size_px' in piecewise_table.colnames
                else inferred_tile_size
                )
            kwargs = {
                'registration_channel': registration_channel,
                'signal_to_control_offset': signal_to_control_offset,
                'coefficient_y': coefficients[frame_i][0],
                'coefficient_x': coefficients[frame_i][1],
                'piecewise_global_shift': (
                    piecewise_table['global_shift_y_px'][frame_i],
                    piecewise_table['global_shift_x_px'][frame_i],
                    ),
                'control_y': control_y,
                'control_x': control_x,
                'tile_size': tile_size,
                }
        masks[mask_i] = registration_valid_mask(
            shape,
            model_used,
            (0, 0) if rigid_translation is None else rigid_translation[frame_i],
            valid_bounds,
            channel=channel,
            **kwargs,
            )
    return masks


def _extract_traces(nwb_path, roi_dict, area_coordinates):
    roi_count = len(roi_dict)
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        preprocessing = nwbfile.processing['preprocessing']
        frame_count = len(preprocessing['registered_signal'].data)
        traces = {
            f'{channel}_{area}_{statistic}': np.full(
                (frame_count, roi_count),
                np.nan,
                dtype=np.float32,
                )
            for channel in ('signal', 'control')
            for area in ('roi', 'surround')
            for statistic in STATISTICS
            }
        bounds = preprocessing['registered_valid_bounds']

        for batch_start in range(0, frame_count, FRAME_BATCH_SIZE):
            batch_stop = min(batch_start + FRAME_BATCH_SIZE, frame_count)
            for channel in ('signal', 'control'):
                frames = np.asarray(
                    preprocessing[f'registered_{channel}'].data[
                        batch_start:batch_stop
                        ]
                    ).swapaxes(1, 2)
                channel_bounds = np.column_stack([
                    np.asarray(
                        bounds[f'{channel}_{coordinate}'][
                            batch_start:batch_stop])
                    for coordinate in ('y0', 'y1', 'x0', 'x1')
                    ])
                valid_masks = _registration_valid_masks(
                    preprocessing,
                    range(batch_start, batch_stop),
                    channel,
                    frames.shape[1:],
                    )
                for area in ('roi', 'surround'):
                    for roi_index, (ypix, xpix) in enumerate(
                            area_coordinates[area]):
                        measurements = _measure_pixels(
                            frames,
                            channel_bounds,
                            ypix,
                            xpix,
                            valid_masks=valid_masks,
                            )
                        for statistic_index, statistic in enumerate(STATISTICS):
                            traces[f'{channel}_{area}_{statistic}'][
                                batch_start:batch_stop,
                                roi_index,
                                ] = measurements[:, statistic_index]
    return traces


#%% NWB storage
def _fluorescence_runs(nwbfile):
    if 'fluorescence' not in nwbfile.processing:
        return {}

    table = nwbfile.processing['fluorescence']['fluorescence_runs']
    runs = {}
    for row_index in range(len(table)):
        run = {
            column_name: table[column_name][row_index]
            for column_name in table.colnames
            }
        for column_name in (
                'run_name', 'roi_run', 'surround_method', 'created_at'):
            run[column_name] = _read_text(run[column_name])
        runs[run['run_name']] = run
    return runs


def list_fluorescence_runs(nwb_path):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        return list(_fluorescence_runs(io.read()).values())


def _check_new_fluorescence_run(nwbfile, run_name):
    if not run_name or '/' in run_name:
        raise ValueError('run_name must be non-empty and cannot contain /')
    if run_name == 'fluorescence_runs':
        raise ValueError('run_name is reserved for fluorescence provenance')
    if run_name in _fluorescence_runs(nwbfile):
        raise ValueError(f'fluorescence run already exists: {run_name}')


def _fluorescence_module(nwbfile):
    if 'fluorescence' in nwbfile.processing:
        return nwbfile.processing['fluorescence']

    module = nwbfile.create_processing_module(
        name='fluorescence',
        description='immutable fluorescence extractions from named ROI runs',
        )
    runs = DynamicTable(
        name='fluorescence_runs',
        description='one provenance row for each immutable fluorescence run',
        )
    columns = {
        'run_name': 'unique immutable extraction run name',
        'roi_run': 'immutable segmentation or curation run used for extraction',
        'surround_method': 'adaptive growth or fixed Euclidean annulus',
        'surround_inner_px': 'excluded distance from each ROI in pixels',
        'surround_outer_px': 'outer distance for fixed surrounds; NaN for adaptive',
        'surround_min_pixels': 'adaptive target pixel count; -1 for fixed',
        'surround_expansion_px': 'adaptive growth step in pixels; -1 for fixed',
        'created_at': 'UTC creation time',
        }
    for name, description in columns.items():
        runs.add_column(name=name, description=description)
    module.add(runs)
    return module


def _add_fluorescence_run(nwbfile, run_metadata, traces):
    run_name = run_metadata['run_name']
    roi_run = run_metadata['roi_run']
    module = _fluorescence_module(nwbfile)
    plane_segmentation = nwbfile.processing['segmentation'][
        'ImageSegmentation'][roi_run]
    roi_rows = list(range(len(plane_segmentation)))
    paired_frames = nwbfile.processing['preprocessing']['paired_frames']
    fluorescence = Fluorescence(name=run_name)

    for series_name, trace_values in traces.items():
        channel, area, statistic = series_name.split('_', 2)
        unit = 'dimensionless' if statistic == 'valid_fraction' else 'counts'
        rois = plane_segmentation.create_roi_table_region(
            region=roi_rows,
            description=f'ROIs from segmentation run {roi_run}',
            )
        fluorescence.add_roi_response_series(RoiResponseSeries(
            name=series_name,
            data=H5DataIO(
                trace_values,
                chunks=(
                    min(1024, len(trace_values)),
                    trace_values.shape[1],
                    ),
                compression='gzip',
                compression_opts=1,
                shuffle=True,
                fletcher32=True,
                ),
            unit=unit,
            rois=rois,
            timestamps=paired_frames,
            description=f'{statistic} {channel} intensity in each {area}',
            ))
    module.add(fluorescence)
    module['fluorescence_runs'].add_row(**run_metadata)


def load_fluorescence_run(nwb_path, run_name):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        nwbfile = io.read()
        runs = _fluorescence_runs(nwbfile)
        if run_name not in runs:
            raise KeyError(f'fluorescence run does not exist: {run_name}')

        fluorescence = nwbfile.processing['fluorescence'][run_name]
        traces = {
            name: np.asarray(series.data).copy()
            for name, series in fluorescence.roi_response_series.items()
            }
        first_series = next(iter(fluorescence.roi_response_series.values()))
        roi_ids = np.asarray(first_series.rois.table.id)[
            np.asarray(first_series.rois.data)
            ].astype(np.int64)
        return {
            'run_name': run_name,
            'roi_run': runs[run_name]['roi_run'],
            'roi_ids': roi_ids,
            'traces': traces,
            'provenance': runs[run_name],
            }


def _verify_fluorescence_run(
        path,
        run_name,
        roi_ids,
        traces,
        registered_movie_state,
        ):
    loaded = load_fluorescence_run(path, run_name)
    np.testing.assert_array_equal(loaded['roi_ids'], roi_ids)
    if set(loaded['traces']) != set(traces):
        raise AssertionError('stored fluorescence series changed')
    frame_count = len(next(iter(traces.values())))
    frame_indices = sorted({0, frame_count // 2, frame_count - 1})
    for series_name, expected_values in traces.items():
        stored_values = loaded['traces'][series_name]
        if stored_values.shape != expected_values.shape:
            raise AssertionError(f'stored fluorescence shape changed: {series_name}')
        np.testing.assert_array_equal(stored_values[frame_indices], expected_values[frame_indices])
    if _registered_movie_state(path) != registered_movie_state:
        raise AssertionError('registered movie storage or seeded frames changed')


#%% public extraction
def extract_fluorescence(
        nwb_path,
        run_name,
        roi_run,
        *,
        surround_method=SURROUND_METHOD,
        surround_inner_px=SURROUND_INNER_PX,
        surround_outer_px=SURROUND_OUTER_PX,
        surround_min_pixels=SURROUND_MIN_PIXELS,
        ):
    if surround_method not in ('adaptive', 'fixed'):
        raise ValueError("surround_method must be 'adaptive' or 'fixed'")
    if surround_inner_px < 0:
        raise ValueError('surround_inner_px must be non-negative')
    if surround_method == 'fixed' and surround_outer_px <= surround_inner_px:
        raise ValueError('fixed surround radii must satisfy inner < outer')
    if surround_method == 'adaptive' and surround_min_pixels < 0:
        raise ValueError('surround_min_pixels must be non-negative')

    nwb_path = Path(nwb_path)
    partial_path = nwb_path.with_name(f'{nwb_path.stem}.extracting.partial.nwb')
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        _check_new_fluorescence_run(nwbfile, run_name)
        if roi_run not in _segmentation_runs(nwbfile):
            raise ValueError(f'ROI run does not exist: {roi_run}')

    loaded_rois = load_roi_run(nwb_path, roi_run)
    roi_dict = loaded_rois['roi_dict']
    if not roi_dict:
        raise ValueError(f'ROI run contains no ROIs: {roi_run}')
    area_coordinates = _roi_and_surround_coordinates(
        roi_dict,
        loaded_rois['reference'].shape,
        surround_method,
        surround_inner_px,
        surround_outer_px,
        surround_min_pixels,
        )
    extraction_start = perf_counter()
    traces = _extract_traces(nwb_path, roi_dict, area_coordinates)
    extraction_time_s = perf_counter() - extraction_start

    run_metadata = {
        'run_name': run_name,
        'roi_run': roi_run,
        'surround_method': surround_method,
        'surround_inner_px': surround_inner_px,
        'surround_outer_px': (
            float(surround_outer_px) if surround_method == 'fixed' else np.nan
            ),
        'surround_min_pixels': (
            surround_min_pixels if surround_method == 'adaptive' else -1
            ),
        'surround_expansion_px': (
            SURROUND_EXPANSION_PX if surround_method == 'adaptive' else -1
            ),
        'created_at': datetime.now(timezone.utc).isoformat(),
        }
    original_size = nwb_path.stat().st_size
    registered_movie_state = _registered_movie_state(nwb_path)
    copy_start = perf_counter()
    shutil.copyfile(nwb_path, partial_path)
    copy_time_s = perf_counter() - copy_start

    with NWBHDF5IO(partial_path, 'a') as io:
        nwbfile = io.read()
        _check_new_fluorescence_run(nwbfile, run_name)
        _add_fluorescence_run(nwbfile, run_metadata, traces)
        io.write(nwbfile)

    validation_errors = [str(error) for error in validate(path=partial_path)]
    if validation_errors:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    _verify_fluorescence_run(
        partial_path,
        run_name,
        np.asarray(list(roi_dict), dtype=np.int64),
        traces,
        registered_movie_state,
        )
    with partial_path.open('rb') as file:
        os.fsync(file.fileno())
    file_size_bytes = partial_path.stat().st_size
    os.replace(partial_path, nwb_path)
    return {
        'output_path': nwb_path,
        'run_name': run_name,
        'roi_run': roi_run,
        'roi_count': len(roi_dict),
        'frame_count': len(next(iter(traces.values()))),
        'extraction_time_s': extraction_time_s,
        'copy_time_s': copy_time_s,
        'file_size_bytes': file_size_bytes,
        'file_size_increase_bytes': file_size_bytes - original_size,
        'validation_errors': validation_errors,
        'peak_memory_bytes': _peak_memory_bytes(),
        }

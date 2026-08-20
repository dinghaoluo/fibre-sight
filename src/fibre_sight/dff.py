'''
Created on 20 August 2026

calculate immutable dF/F runs from extracted fluorescence

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
import pandas as pd
from pynwb import NWBHDF5IO, validate
from pynwb.ophys import Fluorescence, RoiResponseSeries

from .fluorescence import (
    _fluorescence_runs,
    _read_text,
    load_fluorescence_run,
    )
from .nwb_segmentation import _peak_memory_bytes, _registered_movie_state


#%% defaults
STATISTIC = 'mean'
BASELINE_PERCENTILE = 20
BASELINE_WINDOW_S = 300
SURROUND_COEFFICIENT = 0.7
CONTROL_CORRECTION = 'none'
BASELINE_METHOD = 'centred_rolling_percentile_reflect'
QUALITY_CONTROL_METHOD = 'analysis_valid_nan_exclusion'
SURROUND_CORRECTION_METHOD = 'raw_subtraction'


#%% calculation
def _rolling_percentile(values, window_frames, percentile):
    padding = window_frames // 2
    padded_values = np.pad(
        values,
        ((padding, padding), (0, 0)),
        mode='reflect',
        )
    baselines = (
        pd.DataFrame(padded_values)
        .rolling(window_frames, min_periods=1)
        .quantile(percentile / 100)
        .to_numpy()
        )
    start = window_frames - 1
    return baselines[start:start + len(values)]


def _dff_from_raw(raw_fluorescence, analysis_valid, window_frames, percentile):
    fluorescence = np.asarray(raw_fluorescence, dtype=np.float32).copy()
    fluorescence[~analysis_valid] = np.nan
    baseline = _rolling_percentile(fluorescence, window_frames, percentile)
    return ((fluorescence - baseline) / baseline).astype(np.float32)


def _calculate_dff_traces(
        fluorescence_traces,
        statistic,
        analysis_valid,
        window_frames,
        baseline_percentile,
        surround_coefficient,
        control_correction,
        ):
    traces = {}
    for channel in ('signal', 'control'):
        roi = fluorescence_traces[f'{channel}_roi_{statistic}']
        surround = fluorescence_traces[f'{channel}_surround_{statistic}']
        corrected = roi - surround_coefficient * surround
        for area, raw_fluorescence in (
                ('roi', roi),
                ('surround', surround),
                ('surround_corrected', corrected),
                ):
            traces[f'{channel}_{area}_dff'] = _dff_from_raw(
                raw_fluorescence,
                analysis_valid,
                window_frames,
                baseline_percentile,
                )

    if control_correction == 'subtract_dff':
        traces['signal_control_corrected_dff'] = (
            traces['signal_surround_corrected_dff']
            - traces['control_surround_corrected_dff']
            )
    return traces


#%% NWB storage
def _dff_runs(nwbfile):
    if 'dff' not in nwbfile.processing:
        return {}

    table = nwbfile.processing['dff']['dff_runs']
    text_columns = {
        'run_name',
        'fluorescence_run',
        'roi_run',
        'statistic',
        'baseline_method',
        'surround_correction',
        'control_correction',
        'quality_control_method',
        'created_at',
        }
    runs = {}
    for row_index in range(len(table)):
        run = {
            column_name: table[column_name][row_index]
            for column_name in table.colnames
            }
        for column_name in text_columns:
            run[column_name] = _read_text(run[column_name])
        runs[run['run_name']] = run
    return runs


def _check_new_dff_run(nwbfile, run_name):
    if not run_name or '/' in run_name:
        raise ValueError('run_name must be non-empty and cannot contain /')
    if run_name == 'dff_runs':
        raise ValueError('run_name is reserved for dF/F provenance')
    if run_name in _dff_runs(nwbfile):
        raise ValueError(f'dF/F run already exists: {run_name}')


def _dff_module(nwbfile):
    if 'dff' in nwbfile.processing:
        return nwbfile.processing['dff']

    module = nwbfile.create_processing_module(
        name='dff',
        description='immutable dF/F calculations from named fluorescence runs',
        )
    runs = DynamicTable(
        name='dff_runs',
        description='one provenance row for each immutable dF/F run',
        )
    columns = {
        'run_name': 'unique immutable dF/F run name',
        'fluorescence_run': 'immutable fluorescence extraction used as input',
        'roi_run': 'immutable segmentation or curation run used for extraction',
        'statistic': 'raw pixel statistic used for dF/F',
        'baseline_method': 'method used to calculate each baseline',
        'baseline_percentile': 'percentile used for each baseline',
        'baseline_window_s': 'centred baseline window in seconds',
        'baseline_window_frames': 'centred baseline window in frames',
        'surround_correction': 'method used to subtract surrounding fluorescence',
        'surround_coefficient': 'fixed raw-space surround subtraction coefficient',
        'control_correction': 'none or signal dF/F minus control dF/F',
        'quality_control_method': 'handling of frames rejected by registration QC',
        'created_at': 'UTC creation time',
        }
    for name, description in columns.items():
        runs.add_column(name=name, description=description)
    module.add(runs)
    return module


def _add_dff_run(nwbfile, run_metadata, traces):
    run_name = run_metadata['run_name']
    source_fluorescence = nwbfile.processing['fluorescence'][
        run_metadata['fluorescence_run']]
    source_series = next(iter(source_fluorescence.roi_response_series.values()))
    roi_rows = np.asarray(source_series.rois.data).tolist()
    plane_segmentation = source_series.rois.table
    paired_frames = nwbfile.processing['preprocessing']['paired_frames']
    dff_container = Fluorescence(name=run_name)

    for series_name, trace_values in traces.items():
        rois = plane_segmentation.create_roi_table_region(
            region=roi_rows,
            description=f'ROIs from segmentation run {run_metadata["roi_run"]}',
            )
        dff_container.add_roi_response_series(RoiResponseSeries(
            name=series_name,
            data=H5DataIO(
                trace_values,
                chunks=(
                    min(1024, len(trace_values)),
                    max(1, trace_values.shape[1]),
                    ),
                compression='gzip',
                compression_opts=1,
                shuffle=True,
                fletcher32=True,
                ),
            unit='dimensionless',
            rois=rois,
            timestamps=paired_frames,
            description=series_name.replace('_', ' '),
            ))
    module = _dff_module(nwbfile)
    module.add(dff_container)
    module['dff_runs'].add_row(**run_metadata)


def load_dff_run(nwb_path, run_name):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        nwbfile = io.read()
        runs = _dff_runs(nwbfile)
        if run_name not in runs:
            raise KeyError(f'dF/F run does not exist: {run_name}')

        dff_container = nwbfile.processing['dff'][run_name]
        traces = {
            name: np.asarray(series.data).copy()
            for name, series in dff_container.roi_response_series.items()
            }
        first_series = next(iter(dff_container.roi_response_series.values()))
        roi_ids = np.asarray(first_series.rois.table.id)[
            np.asarray(first_series.rois.data)
            ].astype(np.int64)
        return {
            'run_name': run_name,
            'fluorescence_run': runs[run_name]['fluorescence_run'],
            'roi_run': runs[run_name]['roi_run'],
            'roi_ids': roi_ids,
            'traces': traces,
            'provenance': runs[run_name],
            }


def _verify_dff_run(
        path,
        run_name,
        roi_ids,
        traces,
        registered_movie_state,
        ):
    loaded = load_dff_run(path, run_name)
    np.testing.assert_array_equal(loaded['roi_ids'], roi_ids)
    if set(loaded['traces']) != set(traces):
        raise AssertionError('stored dF/F series changed')
    frame_indices = sorted({
        0,
        len(next(iter(traces.values()))) // 2,
        len(next(iter(traces.values()))) - 1,
        })
    for series_name, expected_values in traces.items():
        stored_values = loaded['traces'][series_name]
        if stored_values.shape != expected_values.shape:
            raise AssertionError(f'stored dF/F shape changed: {series_name}')
        np.testing.assert_array_equal(
            stored_values[frame_indices], expected_values[frame_indices])
    if _registered_movie_state(path) != registered_movie_state:
        raise AssertionError('registered movie storage or seeded frames changed')


#%% public calculation
def calculate_dff(
        nwb_path,
        run_name,
        fluorescence_run,
        *,
        statistic=STATISTIC,
        baseline_percentile=BASELINE_PERCENTILE,
        baseline_window_s=BASELINE_WINDOW_S,
        surround_coefficient=SURROUND_COEFFICIENT,
        control_correction=CONTROL_CORRECTION,
        ):
    if statistic not in ('mean', 'median'):
        raise ValueError("statistic must be 'mean' or 'median'")
    if not 0 <= baseline_percentile <= 100:
        raise ValueError('baseline_percentile must be between 0 and 100')
    if baseline_window_s <= 0:
        raise ValueError('baseline_window_s must be positive')
    if control_correction not in ('none', 'subtract_dff'):
        raise ValueError("control_correction must be 'none' or 'subtract_dff'")

    nwb_path = Path(nwb_path)
    partial_path = nwb_path.with_name(
        f'{nwb_path.stem}.calculating_dff.partial.nwb')
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        _check_new_dff_run(nwbfile, run_name)
        fluorescence_runs = _fluorescence_runs(nwbfile)
        if fluorescence_run not in fluorescence_runs:
            raise ValueError(
                f'fluorescence run does not exist: {fluorescence_run}')
        sampling_frequency_hz = float(
            nwbfile.processing['preprocessing'][
                'recording_metadata']['sampling_frequency_hz'][0]
            )
        analysis_valid = np.asarray(
            nwbfile.processing['quality_control'][
                'registration_qc']['analysis_valid'],
            dtype=bool,
            )

    baseline_window_frames = int(round(
        baseline_window_s * sampling_frequency_hz))
    if baseline_window_frames < 1:
        raise ValueError('baseline window is shorter than one frame')

    extracted = load_fluorescence_run(nwb_path, fluorescence_run)
    calculation_start = perf_counter()
    traces = _calculate_dff_traces(
        extracted['traces'],
        statistic,
        analysis_valid,
        baseline_window_frames,
        baseline_percentile,
        surround_coefficient,
        control_correction,
        )
    calculation_time_s = perf_counter() - calculation_start
    run_metadata = {
        'run_name': run_name,
        'fluorescence_run': fluorescence_run,
        'roi_run': extracted['roi_run'],
        'statistic': statistic,
        'baseline_method': BASELINE_METHOD,
        'baseline_percentile': float(baseline_percentile),
        'baseline_window_s': float(baseline_window_s),
        'baseline_window_frames': baseline_window_frames,
        'surround_correction': SURROUND_CORRECTION_METHOD,
        'surround_coefficient': float(surround_coefficient),
        'control_correction': control_correction,
        'quality_control_method': QUALITY_CONTROL_METHOD,
        'created_at': datetime.now(timezone.utc).isoformat(),
        }

    original_size = nwb_path.stat().st_size
    registered_movie_state = _registered_movie_state(nwb_path)
    copy_start = perf_counter()
    shutil.copyfile(nwb_path, partial_path)
    copy_time_s = perf_counter() - copy_start

    with NWBHDF5IO(partial_path, 'a') as io:
        nwbfile = io.read()
        _check_new_dff_run(nwbfile, run_name)
        _add_dff_run(nwbfile, run_metadata, traces)
        io.write(nwbfile)

    validation_errors = [str(error) for error in validate(path=partial_path)]
    if validation_errors:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    _verify_dff_run(
        partial_path,
        run_name,
        extracted['roi_ids'],
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
        'fluorescence_run': fluorescence_run,
        'roi_run': extracted['roi_run'],
        'roi_count': len(extracted['roi_ids']),
        'frame_count': len(next(iter(traces.values()))),
        'calculation_time_s': calculation_time_s,
        'copy_time_s': copy_time_s,
        'file_size_bytes': file_size_bytes,
        'file_size_increase_bytes': file_size_bytes - original_size,
        'validation_errors': validation_errors,
        'peak_memory_bytes': _peak_memory_bytes(),
        }

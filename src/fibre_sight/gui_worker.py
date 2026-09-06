'''
Created on 21 August 2026

run the automatic recording stages outside the Qt event loop

@author: Dinghao Luo
'''

#%% imports
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from pynwb import NWBHDF5IO

from .api import (
    BUNDLED_CHECKPOINT,
    BUNDLED_MIN_SIZE,
    BUNDLED_THRESHOLD,
    calculate_dff,
    extract_fluorescence,
    list_dff_runs,
    list_fluorescence_runs,
    list_roi_runs,
    preprocess_recording,
    segment_recording,
    )
from .dff import (
    BASELINE_PERCENTILE,
    BASELINE_WINDOW_S,
    CONTROL_CORRECTION,
    STATISTIC,
    SURROUND_COEFFICIENT,
    )
from .fluorescence import (
    SURROUND_INNER_PX,
    SURROUND_METHOD,
    SURROUND_MIN_PIXELS,
    SURROUND_OUTER_PX,
    )
from .nwb_segmentation import _checkpoint_sha256
from .preprocessing import SEGMENTATION_REFERENCE_PERCENTILES
from . import __version__


#%% session record
SESSION_SCHEMA_VERSION = 1
_NUMBER = re.compile(r'(\d+)')


def natural_tiff_paths(directory):
    directory = Path(directory)
    paths = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in {'.tif', '.tiff'}
        ]

    def sort_key(path):
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in _NUMBER.split(path.name)
            )

    return sorted(paths, key=sort_key)


def fingerprint_paths(paths):
    records = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        records.append({
            'path': str(path),
            'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            })
    return records


def append_session_event(log_path, event):
    event = {
        'time': datetime.now(timezone.utc).isoformat(),
        **event,
        }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, sort_keys=True, default=str) + '\n').encode()
    mode = 'r+b' if log_path.exists() else 'w+b'
    with log_path.open(mode) as file:
        file.seek(0, os.SEEK_END)
        if file.tell():
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b'\n':
                file.seek(0)
                contents = file.read()
                line_start = contents.rfind(b'\n') + 1
                try:
                    json.loads(contents[line_start:])
                except (UnicodeDecodeError, json.JSONDecodeError):
                    file.seek(line_start)
                    file.truncate()
                else:
                    file.seek(0, os.SEEK_END)
                    file.write(b'\n')
        file.seek(0, os.SEEK_END)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def read_session_events(log_path):
    lines = Path(log_path).read_text(encoding='utf-8').splitlines()
    events = []
    for line_i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if line_i != len(lines) - 1:
                raise
    return events


def latest_session_config(log_path):
    configurations = [
        event['config'] for event in read_session_events(log_path)
        if event.get('event') == 'configured'
        ]
    if not configurations:
        raise ValueError('session log does not contain a configuration')
    return configurations[-1]


def session_stage_states(log_path):
    states = {}
    for event in read_session_events(log_path):
        stage = event.get('stage')
        if stage:
            states[stage] = event['event']
    return states


def partial_paths(config):
    nwb_path = Path(config['output_path'])
    return [
        nwb_path.with_suffix('.partial.nwb'),
        nwb_path.with_name(f'{nwb_path.stem}.segmenting.partial.nwb'),
        nwb_path.with_name(f'{nwb_path.stem}.extracting.partial.nwb'),
        nwb_path.with_name(f'{nwb_path.stem}.calculating_dff.partial.nwb'),
        ]


def _verify_inputs(config):
    records = [
        *config['source_files'],
        config['segmentation']['checkpoint_file'],
        ]
    for record in records:
        path = Path(record['path'])
        stat = path.stat()
        if (
                stat.st_size != record['size_bytes']
                or stat.st_mtime_ns != record['mtime_ns']
                ):
            raise ValueError(f'input file changed since configuration: {path}')


#%% completed-stage checks
def _preprocessing_complete(config):
    path = Path(config['output_path'])
    if not path.exists():
        return False
    with NWBHDF5IO(path, 'r') as io:
        nwbfile = io.read()
        if (
                'preprocessing' not in nwbfile.processing
                or 'quality_control' not in nwbfile.processing
                ):
            return False
        preprocessing = nwbfile.processing['preprocessing']
        metadata = preprocessing['recording_metadata']
        model = preprocessing['registration_model']
        reference = preprocessing['segmentation_reference_metadata']

        def text(value):
            return value.decode() if isinstance(value, bytes) else str(value)

        source = config['source']
        registration = config['registration']
        segmentation = config['segmentation']
        pixel_size_um = float(metadata['pixel_size_um'][0])
        expected_pixel_size = source['pixel_size_um']
        pixel_size_matches = (
            (expected_pixel_size is None and pixel_size_um != pixel_size_um)
            or (
                expected_pixel_size is not None
                and pixel_size_um == expected_pixel_size
                )
            )
        matches = (
            float(metadata['sampling_frequency_hz'][0])
            == source['sampling_frequency_hz']
            and bool(metadata['multiplexed'][0]) == source['multiplexed']
            and int(metadata['signal_channel'][0]) == source['signal_channel']
            and int(metadata['control_channel'][0]) == source['control_channel']
            and text(metadata['signal_label'][0]) == source['signal_label']
            and text(metadata['control_label'][0]) == source['control_label']
            and pixel_size_matches
            and text(model['requested_model'][0]) == registration['model']
            and text(model['registration_channel'][0]) == registration['channel']
            and text(reference['source_channel'][0])
            == segmentation['reference_channel']
            and float(reference['low_percentile'][0])
            == segmentation['reference_low_percentile']
            and float(reference['high_percentile'][0])
            == segmentation['reference_high_percentile']
            )
    if not matches:
        raise ValueError('preprocessed NWB exists with different parameters')
    return True


def _proposal_complete(config):
    expected = config['segmentation']
    runs = {
        run['run_name']: run
        for run in list_roi_runs(config['output_path'], run_type='proposed')
        }
    run = runs.get(expected['run_name'])
    if run is None:
        return False
    matches = (
        float(run['threshold']) == expected['threshold']
        and int(run['min_size']) == expected['min_size']
        and bool(run['tta']) == expected['tta']
        and Path(run['checkpoint_path']).resolve()
        == Path(expected['checkpoint_path']).resolve()
        and run['checkpoint_sha256']
        == _checkpoint_sha256(expected['checkpoint_path'])
        )
    if not matches:
        run_name = expected['run_name']
        raise ValueError(
            f'ROI run exists with different parameters: {run_name}')
    return True


def _fluorescence_complete(config):
    expected = config['extraction']
    runs = {
        run['run_name']: run
        for run in list_fluorescence_runs(config['output_path'])
        }
    run = runs.get(expected['run_name'])
    if run is None:
        return False
    matches = (
        run['roi_run'] == expected['roi_run']
        and run['surround_method'] == expected['surround_method']
        and float(run['surround_inner_px']) == expected['surround_inner_px']
        )
    if expected['surround_method'] == 'fixed':
        matches = (
            matches
            and float(run['surround_outer_px']) == expected['surround_outer_px']
            )
    else:
        matches = (
            matches
            and int(run['surround_min_pixels']) == expected['surround_min_pixels']
            )
    if not matches:
        raise ValueError(
            'fluorescence run exists with different parameters: '
            f'{expected["run_name"]}')
    return True


def _dff_complete(config):
    expected = config['dff']
    runs = {
        run['run_name']: run
        for run in list_dff_runs(config['output_path'])
        }
    run = runs.get(expected['run_name'])
    if run is None:
        return False
    matches = (
        run['fluorescence_run'] == expected['fluorescence_run']
        and run['statistic'] == expected['statistic']
        and float(run['baseline_percentile']) == expected['baseline_percentile']
        and float(run['baseline_window_s']) == expected['baseline_window_s']
        and float(run['surround_coefficient']) == expected['surround_coefficient']
        and run['control_correction'] == expected['control_correction']
        )
    if not matches:
        raise ValueError(
            f'dF/F run exists with different parameters: {expected["run_name"]}')
    return True


#%% stages
def _stage_start_text(stage, config):
    if stage == 'preprocessing':
        signal_count = len(config['signal_files'])
        control_count = len(config['control_files'])
        if config['source']['multiplexed']:
            source_text = f'{signal_count:,} multiplexed TIFF files'
        else:
            source_text = (
                f'{signal_count:,} signal and '
                f'{control_count:,} control TIFF files'
                )
        return (
            f'preprocessing: registering {source_text} and building channel '
            'references'
            )
    if stage == 'segmentation':
        threshold = config['segmentation']['threshold']
        return (
            'segmentation: running the bundled model on the stored reference '
            f'at threshold {threshold:.2f}'
            )
    if stage == 'extraction':
        surround_method = config['extraction']['surround_method']
        return (
            'extraction: measuring ROI and surround fluorescence across the '
            f'recording ({surround_method} surround)'
            )
    baseline_window = config['dff']['baseline_window_s']
    return (
        'dF/F: calculating baseline-normalised traces over '
        f'{baseline_window:.1f} s'
        )


def _stage_display_name(stage):
    return 'dF/F' if stage == 'dff' else stage


def _stage_result_text(stage, result):
    if stage == 'preprocessing':
        frame_count = result.get('n_frames', 'unknown')
        reference_count = result.get('segmentation_reference_frames', '?')
        selected_model = result.get('selected_model', 'unknown')
        return (
            f'{frame_count} paired frames; '
            f'{reference_count} reference frames; '
            f'{selected_model} registration'
            )
    if stage == 'segmentation':
        roi_count = result.get('roi_count', 'unknown')
        return f'{roi_count} ROI proposals'
    if stage == 'extraction':
        roi_count = result.get('roi_count', 'unknown')
        frame_count = result.get('frame_count', '?')
        return (
            f'{roi_count} ROIs measured over '
            f'{frame_count} frames'
            )
    roi_count = result.get('roi_count', 'unknown')
    return f'{roi_count} ROI traces'


def _run_stage(log_path, stage, completed, operation, config=None):
    display_stage = _stage_display_name(stage)
    try:
        if completed():
            append_session_event(log_path, {
                'event': 'stage_skipped',
                'stage': stage,
                'reason': 'matching NWB result already exists',
                })
            print(f'{display_stage}: existing result retained', flush=True)
            return

        append_session_event(log_path, {'event': 'stage_started', 'stage': stage})
        print(
            _stage_start_text(stage, config)
            if config is not None else
            f'{display_stage}: started',
            flush=True,
            )
        result = operation()
    except Exception as exc:
        append_session_event(log_path, {
            'event': 'stage_failed',
            'stage': stage,
            'error': str(exc),
            })
        raise
    append_session_event(log_path, {
        'event': 'stage_completed',
        'stage': stage,
        'result': result,
        })
    print(
        f'{display_stage}: completed; {_stage_result_text(stage, result)}',
        flush=True,
        )


def run_session(log_path):
    log_path = Path(log_path)
    config = latest_session_config(log_path)
    _verify_inputs(config)
    nwb_path = Path(config['output_path'])
    source = config['source']
    registration = config['registration']
    segmentation = config['segmentation']
    extraction = config['extraction']
    dff = config['dff']

    signal_paths = [record['path'] for record in config['signal_files']]
    control_paths = (
        [record['path'] for record in config['control_files']]
        if not source['multiplexed'] else
        None
        )
    _run_stage(
        log_path,
        'preprocessing',
        lambda: _preprocessing_complete(config),
        lambda: preprocess_recording(
            signal_paths,
            nwb_path,
            signal_channel=source['signal_channel'],
            control_channel=source['control_channel'],
            multiplexed=source['multiplexed'],
            sampling_frequency_hz=source['sampling_frequency_hz'],
            signal_label=source['signal_label'],
            control_label=source['control_label'],
            control_tiff_paths=control_paths,
            registration_model=registration['model'],
            registration_channel=registration['channel'],
            pixel_size_um=source['pixel_size_um'],
            segmentation_reference_channel=segmentation['reference_channel'],
            segmentation_reference_percentiles=(
                segmentation['reference_low_percentile'],
                segmentation['reference_high_percentile'],
                ),
            ),
        config,
        )
    _run_stage(
        log_path,
        'segmentation',
        lambda: _proposal_complete(config),
        lambda: segment_recording(
            nwb_path,
            segmentation['run_name'],
            checkpoint_path=segmentation['checkpoint_path'],
            threshold=segmentation['threshold'],
            min_size=segmentation['min_size'],
            tta=segmentation['tta'],
            device=segmentation['device'],
            ),
        config,
        )
    _run_stage(
        log_path,
        'extraction',
        lambda: _fluorescence_complete(config),
        lambda: extract_fluorescence(
            nwb_path,
            extraction['run_name'],
            extraction['roi_run'],
            surround_method=extraction['surround_method'],
            surround_inner_px=extraction['surround_inner_px'],
            surround_outer_px=extraction['surround_outer_px'],
            surround_min_pixels=extraction['surround_min_pixels'],
            ),
        config,
        )
    _run_stage(
        log_path,
        'dff',
        lambda: _dff_complete(config),
        lambda: calculate_dff(
            nwb_path,
            dff['run_name'],
            dff['fluorescence_run'],
            statistic=dff['statistic'],
            baseline_percentile=dff['baseline_percentile'],
            baseline_window_s=dff['baseline_window_s'],
            surround_coefficient=dff['surround_coefficient'],
            control_correction=dff['control_correction'],
            ),
        config,
        )
    append_session_event(log_path, {'event': 'session_completed'})
    print(f'automatic session completed: {nwb_path}', flush=True)


#%% command line
def _direct_session_config(args):
    signal_dir = args.session_path.expanduser().resolve()
    if not signal_dir.is_dir():
        raise ValueError(f'TIFF folder does not exist: {signal_dir}')
    signal_paths = natural_tiff_paths(signal_dir)
    if not signal_paths:
        raise ValueError(f'no TIFF files found in {signal_dir}')

    multiplexed = args.layout == 'interleaved'
    control_paths = []
    if not multiplexed:
        if args.control_tiff_dir is None:
            raise ValueError('--control-tiff-dir is required for separate TIFF folders')
        control_dir = args.control_tiff_dir.expanduser().resolve()
        if not control_dir.is_dir():
            raise ValueError(f'TIFF folder does not exist: {control_dir}')
        control_paths = natural_tiff_paths(control_dir)
        if len(control_paths) != len(signal_paths):
            raise ValueError('signal and control TIFF folders contain different file counts')

    if args.output is None:
        raise ValueError('--output is required when running from a TIFF folder')
    output_path = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f'checkpoint does not exist: {checkpoint_path}')
    low, high = args.reference_low_percentile, args.reference_high_percentile
    if low >= high:
        raise ValueError('reference percentiles must satisfy lower < upper')

    signal_records = fingerprint_paths(signal_paths)
    control_records = fingerprint_paths(control_paths)
    return {
        'schema_version': SESSION_SCHEMA_VERSION,
        'fibre_sight_version': __version__,
        'output_path': str(output_path),
        'source_files': signal_records + control_records,
        'signal_files': signal_records,
        'control_files': control_records,
        'source': {
            'multiplexed': multiplexed,
            'sampling_frequency_hz': args.sampling_frequency,
            'signal_channel': args.signal_channel,
            'control_channel': args.control_channel,
            'signal_label': args.signal_label,
            'control_label': args.control_label,
            'pixel_size_um': args.pixel_size,
            },
        'registration': {
            'model': args.registration_model,
            'channel': args.registration_channel,
            },
        'segmentation': {
            'run_name': args.proposal_run,
            'reference_channel': args.reference_channel,
            'reference_low_percentile': low,
            'reference_high_percentile': high,
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_file': fingerprint_paths([checkpoint_path])[0],
            'threshold': args.threshold,
            'min_size': args.min_size,
            'tta': not args.no_tta,
            'device': args.device,
            },
        'extraction': {
            'run_name': args.fluorescence_run,
            'roi_run': args.roi_run,
            'surround_method': args.surround_method,
            'surround_inner_px': args.surround_inner_px,
            'surround_outer_px': args.surround_outer_px,
            'surround_min_pixels': args.surround_min_pixels,
            },
        'dff': {
            'run_name': args.dff_run,
            'fluorescence_run': args.fluorescence_run,
            'statistic': args.statistic,
            'baseline_percentile': args.baseline_percentile,
            'baseline_window_s': args.baseline_window_s,
            'surround_coefficient': args.surround_coefficient,
            'control_correction': args.control_correction,
            },
        }


def _add_direct_arguments(parser):
    parser.add_argument('--output', type=Path, help='NWB output path')
    parser.add_argument('--control-tiff-dir', type=Path)
    parser.add_argument('--layout', choices=('interleaved', 'separate'), default='interleaved')
    parser.add_argument('--signal-channel', type=int, default=1)
    parser.add_argument('--control-channel', type=int, default=2)
    parser.add_argument('--sampling-frequency', type=float, default=30)
    parser.add_argument('--signal-label', default='signal')
    parser.add_argument('--control-label', default='control')
    parser.add_argument('--pixel-size', type=float)
    parser.add_argument('--registration-model', choices=('rigid', 'piecewise', 'auto'), default='auto')
    parser.add_argument('--registration-channel', choices=('signal', 'control'), default='control')
    parser.add_argument('--reference-channel', choices=('signal', 'control'), default='control')
    parser.add_argument('--reference-low-percentile', type=float, default=SEGMENTATION_REFERENCE_PERCENTILES[0])
    parser.add_argument('--reference-high-percentile', type=float, default=SEGMENTATION_REFERENCE_PERCENTILES[1])
    parser.add_argument('--checkpoint', type=Path, default=BUNDLED_CHECKPOINT)
    parser.add_argument('--threshold', type=float, default=BUNDLED_THRESHOLD)
    parser.add_argument('--min-size', type=int, default=BUNDLED_MIN_SIZE)
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--proposal-run', default='proposal_auto')
    parser.add_argument('--roi-run', default='proposal_auto')
    parser.add_argument('--fluorescence-run', default='fluorescence_auto')
    parser.add_argument('--surround-method', choices=('adaptive', 'fixed'), default=SURROUND_METHOD)
    parser.add_argument('--surround-inner-px', type=int, default=SURROUND_INNER_PX)
    parser.add_argument('--surround-outer-px', type=int, default=SURROUND_OUTER_PX)
    parser.add_argument('--surround-min-pixels', type=int, default=SURROUND_MIN_PIXELS)
    parser.add_argument('--dff-run', default='dff_auto')
    parser.add_argument('--statistic', choices=('mean', 'median'), default=STATISTIC)
    parser.add_argument('--baseline-percentile', type=float, default=BASELINE_PERCENTILE)
    parser.add_argument('--baseline-window-s', type=float, default=BASELINE_WINDOW_S)
    parser.add_argument('--surround-coefficient', type=float, default=SURROUND_COEFFICIENT)
    parser.add_argument('--control-correction', choices=('none', 'subtract_dff'), default=CONTROL_CORRECTION)


def parse_args():
    parser = argparse.ArgumentParser(
        description='run an automatic FibreSight session from TIFFs or a session log',
        )
    parser.add_argument('session_path', type=Path, help='TIFF folder or .fibresight.jsonl log')
    _add_direct_arguments(parser)
    return parser.parse_args()


def run_direct_session(args):
    config = _direct_session_config(args)
    log_path = Path(config['output_path']).with_suffix('.fibresight.jsonl')
    if log_path.exists():
        if latest_session_config(log_path) != config:
            raise ValueError(
                f'session log already exists with different settings: {log_path}')
    else:
        append_session_event(log_path, {'event': 'configured', 'config': config})
    run_session(log_path)


def main():
    args = parse_args()
    try:
        if args.session_path.suffix.casefold() == '.jsonl':
            run_session(args.session_path)
        else:
            run_direct_session(args)
    except Exception as exc:
        print(f'automatic session failed: {exc}', flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()

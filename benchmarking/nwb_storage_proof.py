'''
Created on 19 August 2026

compare float32 and int16 storage for registered NWB movies

@author: Dinghao Luo
'''

#%% imports
import json
from pathlib import Path
import subprocess
import sys

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, validate
from pynwb.base import Images
from pynwb.image import GrayscaleImage, ImageSeries
import tifffile

from fibre_sight.preprocessing import (
    _estimate_shifts,
    _prepare_reference,
    add_quality_control_to_nwb,
    estimate_channel_offset,
    index_tiffs,
    make_reference,
    measure_quality,
    read_session_start_time,
    read_tiffs,
    register_pair,
    warp_frame,
    )


#%% paths and recording window
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / 'workspace' / 'dev' / 'lab-session-3'
FLOAT32_PATH = PROJECT_ROOT / 'workspace' / 'dev' / 'nwb_float32.partial.nwb'
INT16_PATH = PROJECT_ROOT / 'workspace' / 'dev' / 'nwb_int16.partial.nwb'
FRAME_START = 550
N_FRAMES = 200
SAMPLING_FREQUENCY_HZ = 30


#%% registration pass
def load_recording_window(recording, frame_start=FRAME_START, n_frames=N_FRAMES):
    frame_index = recording['frames'][frame_start:frame_start + n_frames]
    if len(frame_index) != n_frames:
        raise ValueError('the requested recording window is incomplete')

    signal = np.empty((n_frames, *recording['shape']), dtype=recording['dtype'])
    control = np.empty_like(signal)
    window = {**recording, 'frames': frame_index, 'n_frames': n_frames}
    for frame_i, pair in enumerate(read_tiffs(window)):
        signal[frame_i] = pair['signal']
        control[frame_i] = pair['control']
    return frame_index, signal, control


def register_window(recording, frame_start=FRAME_START, n_frames=N_FRAMES):
    frame_index, signal, control = load_recording_window(
        recording, frame_start, n_frames)
    reference_info = make_reference(control)
    reference = reference_info['image']
    estimates = _estimate_shifts(_prepare_reference(reference), control)
    timestamps = np.asarray([
        frame['frame'] / recording['sampling_frequency_hz']
        for frame in frame_index
        ])

    alignment_frames = np.linspace(0, n_frames - 1, min(40, n_frames), dtype=int)
    aligned_signal = []
    aligned_control = []
    for frame_i in alignment_frames:
        estimate = estimates[frame_i]
        pair = register_pair(
            signal[frame_i], control[frame_i],
            estimate['shift_y'], estimate['shift_x'])
        aligned_signal.append(pair['signal'])
        aligned_control.append(pair['control'])
    channel_offset = estimate_channel_offset(aligned_signal, aligned_control)

    registered_control = np.empty(control.shape, dtype=np.float32)
    control_bounds = np.empty((n_frames, 4), dtype=np.int32)

    def keep_registered_control(frame_i, frame, bounds):
        registered_control[frame_i] = frame
        control_bounds[frame_i] = bounds

    quality = measure_quality(
        reference,
        control,
        estimates,
        recording['sampling_frequency_hz'],
        timestamps=timestamps,
        write_registered=keep_registered_control,
        )

    registered_signal = np.empty(signal.shape, dtype=np.float32)
    signal_bounds = np.empty((n_frames, 4), dtype=np.int32)
    for frame_i, (frame, estimate) in enumerate(zip(signal, estimates)):
        signal_shift = (
            estimate['shift_y'] + channel_offset['shift_y'],
            estimate['shift_x'] + channel_offset['shift_x'],
            )
        registered_signal[frame_i], signal_bounds[frame_i] = warp_frame(
            frame, *signal_shift)

    return {
        'frame_index': frame_index,
        'timestamps': timestamps,
        'reference': reference,
        'reference_info': reference_info,
        'estimates': estimates,
        'channel_offset': channel_offset,
        'quality': quality,
        'signal': registered_signal,
        'control': registered_control,
        'signal_bounds': signal_bounds,
        'control_bounds': control_bounds,
        }


#%% storage
def int16_movie(movie):
    limits = np.iinfo(np.int16)
    stored = np.zeros(movie.shape, dtype=np.int16)
    clipped = False
    max_error = 0.0
    for frame_i, frame in enumerate(movie):
        valid = np.isfinite(frame)
        rounded = np.rint(frame[valid])
        clipped |= bool(np.any((rounded < limits.min) | (rounded > limits.max)))
        rounded = np.clip(rounded, limits.min, limits.max)
        stored[frame_i][valid] = rounded.astype(np.int16)
        if len(rounded):
            max_error = max(
                max_error, float(np.max(np.abs(frame[valid] - rounded))))
    return stored, clipped, max_error


def add_recording_metadata(
        module,
        recording,
        result,
        session_time_source,
        valid_bounds=None,
        ):
    metadata = DynamicTable(
        name='recording_metadata',
        description='recording-level values used for this storage proof',
        )
    metadata_columns = {
        'sampling_frequency_hz': 'paired observations per second',
        'multiplexed': 'whether signal and control pages alternate in one TIFF',
        'signal_channel': 'one-based signal position within each TIFF pair',
        'control_channel': 'one-based control position within each TIFF pair',
        'signal_label': 'signal-channel label',
        'control_label': 'control-channel label',
        'frame_height': 'image height in pixels',
        'frame_width': 'image width in pixels',
        'source_dtype': 'dtype of the raw TIFF pages',
        'session_start_time_source': 'source used for the required NWB session time',
        'session_start_time_raw': 'unparsed source value',
        'session_start_time_timezone': 'timezone applied to the source value',
        }
    for name, description in metadata_columns.items():
        metadata.add_column(name=name, description=description)
    metadata.add_row(
        sampling_frequency_hz=recording['sampling_frequency_hz'],
        multiplexed=recording['multiplexed'],
        signal_channel=recording['signal_channel'],
        control_channel=recording['control_channel'],
        signal_label=recording['signal_label'],
        control_label=recording['control_label'],
        frame_height=recording['shape'][0],
        frame_width=recording['shape'][1],
        source_dtype=str(recording['dtype']),
        session_start_time_source=session_time_source['source'],
        session_start_time_raw=session_time_source['raw_value'],
        session_start_time_timezone=session_time_source['timezone'],
        )
    module.add(metadata)

    source_tiffs = DynamicTable(
        name='source_tiffs',
        description='raw TIFF files forming the indexed recording',
        )
    source_tiffs.add_column(name='path', description='absolute TIFF path')
    source_tiffs.add_column(
        name='channel_role', description='signal, control, or signal/control')
    source_tiffs.add_column(name='n_pages', description='number of TIFF pages')
    signal_paths = set(recording['signal_tiffs'])
    control_paths = set(recording['control_tiffs'])
    for path in dict.fromkeys(recording['signal_tiffs'] + recording['control_tiffs']):
        roles = []
        if path in signal_paths:
            roles.append('signal')
        if path in control_paths:
            roles.append('control')
        with tifffile.TiffFile(path) as tiff:
            n_pages = len(tiff.pages)
        source_tiffs.add_row(
            path=str(path.resolve()),
            channel_role='/'.join(roles),
            n_pages=n_pages,
            )
    module.add(source_tiffs)

    frame_table = DynamicTable(
        name='recording_index',
        description='TIFF pages for each paired frame stored in this proof',
        )
    for name, description in (
            ('frame', 'zero-based paired frame in the source recording'),
            ('signal_tiff', 'absolute path of the signal TIFF'),
            ('signal_page', 'zero-based signal page within that TIFF'),
            ('control_tiff', 'absolute path of the control TIFF'),
            ('control_page', 'zero-based control page within that TIFF'),
            ):
        frame_table.add_column(name=name, description=description)
    for frame in result['frame_index']:
        frame_table.add_row(
            frame=frame['frame'],
            signal_tiff=str(frame['signal_tiff'].resolve()),
            signal_page=frame['signal_page'],
            control_tiff=str(frame['control_tiff'].resolve()),
            control_page=frame['control_page'],
            )
    module.add(frame_table)

    channel_alignment = DynamicTable(
        name='channel_alignment',
        description='static signal-to-control translation estimated from four time quartiles',
        )
    for name, description in (
            ('dy_px', 'applied signal correction along y'),
            ('dx_px', 'applied signal correction along x'),
            ('candidate_dy_px', 'median candidate correction along y'),
            ('candidate_dx_px', 'median candidate correction along x'),
            ('max_disagreement_px', 'largest quartile disagreement from the candidate'),
            ('gradient_ncc_before', 'mean quartile gradient NCC before alignment'),
            ('gradient_ncc_after', 'mean quartile gradient NCC after alignment'),
            ('accepted', 'whether the candidate met both acceptance boundaries'),
            ):
        channel_alignment.add_column(name=name, description=description)
    offset = result['channel_offset']
    channel_alignment.add_row(
        dy_px=offset['shift_y'],
        dx_px=offset['shift_x'],
        candidate_dy_px=offset['candidate_y'],
        candidate_dx_px=offset['candidate_x'],
        max_disagreement_px=offset['max_disagreement_px'],
        gradient_ncc_before=offset['gradient_ncc_before'],
        gradient_ncc_after=offset['gradient_ncc_after'],
        accepted=offset['accepted'],
        )
    module.add(channel_alignment)

    if valid_bounds is not None:
        bounds_table = DynamicTable(
            name='registered_valid_bounds',
            description=(
                'half-open y/x rectangles after the stored x/y image is returned '
                'to row/column matrix order'),
            )
        for channel in ('signal', 'control'):
            for coordinate in ('y0', 'y1', 'x0', 'x1'):
                bounds_table.add_column(
                    name=f'{channel}_{coordinate}',
                    description=f'{channel} valid {coordinate} (px)',
                    )
        for signal_bounds, control_bounds in zip(
                valid_bounds['signal'], valid_bounds['control']):
            bounds_table.add_row(**{
                **dict(zip(
                    ('signal_y0', 'signal_y1', 'signal_x0', 'signal_x1'),
                    signal_bounds,
                    )),
                **dict(zip(
                    ('control_y0', 'control_y1', 'control_x0', 'control_x1'),
                    control_bounds,
                    )),
                })
        module.add(bounds_table)


def write_storage_file(
        path,
        recording,
        result,
        signal_movie,
        control_movie,
        session_start_time,
        session_time_source,
        valid_bounds=None,
        ):
    n_frames, height, width = signal_movie.shape
    nwbfile = NWBFile(
        session_description='FibreSight NWB storage proof',
        identifier=f'fibre-sight-storage-proof-{signal_movie.dtype}',
        session_start_time=session_start_time,
        timestamps_reference_time=session_start_time,
        )
    module = nwbfile.create_processing_module(
        name='preprocessing',
        description='registered movies and the measurements used to create them',
        )
    paired_frames = TimeSeries(
        name='paired_frames',
        data=np.asarray([frame['frame'] for frame in result['frame_index']]),
        unit='frame',
        timestamps=result['timestamps'],
        description='source recording frame and paired-observation time',
        )
    module.add(paired_frames)

    # 19 August 2026: NWB image axes are x, y; TIFF and NumPy images are row-y, column-x
    for channel, movie in (
            ('signal', signal_movie),
            ('control', control_movie),
            ):
        module.add(ImageSeries(
            name=f'registered_{channel}',
            data=H5DataIO(
                movie.swapaxes(1, 2),
                chunks=(1, width, height),
                compression='gzip',
                compression_opts=1,
                shuffle=True,
                fletcher32=True,
            ),
            unit='counts',
            format='raw',
            dimension=(width, height),
            num_samples=np.uint64(n_frames),
            timestamps=paired_frames,
            description=f'rigidly registered {channel} movie',
            ))

    translations = np.asarray([
        [estimate['shift_y'], estimate['shift_x']]
        for estimate in result['estimates']
        ], dtype=np.float32)
    module.add(TimeSeries(
        name='rigid_translation',
        data=translations,
        unit='pixels',
        timestamps=paired_frames,
        description='applied control translation in dy_px, dx_px order',
        ))
    module.add(Images(
        name='registration_references',
        images=[GrayscaleImage(
            name='control_reference',
            data=result['reference'].T,
            description='two-pass control-channel reference',
            )],
        description='images used to estimate registration',
        ))
    add_recording_metadata(
        module,
        recording,
        result,
        session_time_source,
        valid_bounds=valid_bounds,
        )
    add_quality_control_to_nwb(nwbfile, result['quality'])

    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwbfile)


#%% validation
def metadata_read_time(path):
    code = '''
import sys
from time import perf_counter
from pynwb import NWBHDF5IO

start = perf_counter()
with NWBHDF5IO(sys.argv[1], 'r') as io:
    nwbfile = io.read()
    session_start_time = nwbfile.session_start_time.isoformat()
    sampling_frequency = nwbfile.processing['preprocessing'][
        'recording_metadata']['sampling_frequency_hz'][0]
elapsed = perf_counter() - start
print(f'{elapsed}\t{session_start_time}\t{sampling_frequency}')
'''
    process = subprocess.run(
        [sys.executable, '-c', code, str(path)],
        check=True,
        capture_output=True,
        text=True,
        )
    elapsed, session_start_time, sampling_frequency = process.stdout.strip().split('\t')
    return float(elapsed), session_start_time, float(sampling_frequency)


def stored_movie_checks(path, result, integer=False):
    rng = np.random.default_rng(42)
    random_frames = np.sort(rng.choice(N_FRAMES, 20, replace=False))
    single_frames = (0, 100, N_FRAMES - 1)
    maximum_error = 0.0
    float32_exact = True
    bounds_match = True
    invalid_pixels_are_zero = True
    channel_bounds = {
        'signal': result['signal_bounds'],
        'control': result['control_bounds'],
        }

    with NWBHDF5IO(path, 'r') as io:
        nwbfile = io.read()
        module = nwbfile.processing['preprocessing']
        for channel in ('signal', 'control'):
            data = module[f'registered_{channel}'].data
            height, width = result[channel].shape[1:]
            expected_shape = width, height
            if data.chunks != (1, *expected_shape):
                raise AssertionError(f'unexpected {channel} chunk shape: {data.chunks}')
            if not (
                    data.compression == 'gzip'
                    and data.compression_opts == 1
                    and data.shuffle
                    and data.fletcher32
                    ):
                raise AssertionError(f'unexpected {channel} HDF5 filters')
            for frame_i in single_frames:
                if np.asarray(data[frame_i]).shape != expected_shape:
                    raise AssertionError('single-frame NWB access returned the wrong shape')
            for frame_i in random_frames:
                stored = np.asarray(data[frame_i]).T
                expected = result[channel][frame_i]
                if integer:
                    valid = np.isfinite(expected)
                    difference = np.abs(stored[valid].astype(np.float32) - expected[valid])
                    maximum_error = max(
                        maximum_error,
                        float(difference.max()) if len(difference) else 0.0,
                        )
                    invalid_pixels_are_zero &= bool(np.all(stored[~valid] == 0))
                else:
                    float32_exact &= bool(np.array_equal(
                        stored.view(np.uint32), expected.view(np.uint32)))

        signal_time = module['registered_signal'].timestamps
        control_time = module['registered_control'].timestamps
        if signal_time is not control_time:
            raise AssertionError('registered movies do not share one timestamps dataset')

        if integer:
            table = module['registered_valid_bounds']
            for channel in ('signal', 'control'):
                saved = np.column_stack([
                    table[f'{channel}_{coordinate}'][:]
                    for coordinate in ('y0', 'y1', 'x0', 'x1')
                    ])
                bounds_match &= bool(np.array_equal(saved, channel_bounds[channel]))

    return {
        'random_frames': random_frames.tolist(),
        'single_frames': list(single_frames),
        'maximum_error_counts': maximum_error,
        'float32_bitwise_exact': float32_exact,
        'valid_bounds_exact': bounds_match,
        'invalid_pixels_are_zero': invalid_pixels_are_zero,
        }


def roi_mean_error(path, result):
    rng = np.random.default_rng(42)
    height, width = result['signal'].shape[1:]
    centres = np.column_stack([
        rng.integers(10, height - 10, 5),
        rng.integers(10, width - 10, 5),
        ])
    y, x = np.indices((height, width))
    rois = [
        (y - centre_y) ** 2 + (x - centre_x) ** 2 <= 10 ** 2
        for centre_y, centre_x in centres
        ]
    maximum_error = 0.0
    with NWBHDF5IO(path, 'r') as io:
        module = io.read().processing['preprocessing']
        for channel in ('signal', 'control'):
            data = module[f'registered_{channel}'].data
            for frame_i, expected in enumerate(result[channel]):
                stored = np.asarray(data[frame_i], dtype=np.float32).T
                valid = np.isfinite(expected)
                for roi in rois:
                    use_pixels = roi & valid
                    if use_pixels.any():
                        maximum_error = max(
                            maximum_error,
                            abs(float(stored[use_pixels].mean() - expected[use_pixels].mean())),
                            )
    return maximum_error, centres.tolist()


def restored_control(path, result, integer=False):
    with NWBHDF5IO(path, 'r') as io:
        module = io.read().processing['preprocessing']
        movie = np.asarray(
            module['registered_control'].data[:], dtype=np.float32).swapaxes(1, 2)
        if integer:
            bounds = result['control_bounds']
            for frame, (y0, y1, x0, x1) in zip(movie, bounds):
                valid = np.zeros(frame.shape, dtype=bool)
                valid[y0:y1, x0:x1] = True
                frame[~valid] = np.nan
    return movie


def stored_qc_states(path, result, integer=False):
    movie = restored_control(path, result, integer)
    estimates = [{
        'shift_y': 0,
        'shift_x': 0,
        'peak_ratio': 2,
        'tile_disagreement': 0,
        'out_of_range': False,
        'search_boundary': False,
        }] * len(movie)
    quality = measure_quality(
        result['reference'],
        movie,
        estimates,
        SAMPLING_FREQUENCY_HZ,
        timestamps=result['timestamps'],
        )
    return quality['recommended_state']


def run_storage_proof():
    raw_tiffs = sorted(RAW_ROOT.glob('*.tif'))
    recording = index_tiffs(
        raw_tiffs,
        signal_channel=1,
        control_channel=2,
        sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
        signal_label='dLight',
        control_label='tdTomato',
        )
    session_start_time, session_time_source = read_session_start_time(
        recording['signal_tiffs'][0])
    result = register_window(recording)

    write_storage_file(
        FLOAT32_PATH,
        recording,
        result,
        result['signal'],
        result['control'],
        session_start_time,
        session_time_source,
        )
    signal_int16, signal_clipped, signal_rounding_error = int16_movie(result['signal'])
    control_int16, control_clipped, control_rounding_error = int16_movie(result['control'])
    write_storage_file(
        INT16_PATH,
        recording,
        result,
        signal_int16,
        control_int16,
        session_start_time,
        session_time_source,
        valid_bounds={
            'signal': result['signal_bounds'],
            'control': result['control_bounds'],
            },
        )

    validation_errors = {
        'float32': [str(error) for error in validate(path=FLOAT32_PATH)],
        'int16': [str(error) for error in validate(path=INT16_PATH)],
        }
    metadata_times = {
        'float32': metadata_read_time(FLOAT32_PATH),
        'int16': metadata_read_time(INT16_PATH),
        }
    float32_checks = stored_movie_checks(FLOAT32_PATH, result)
    int16_checks = stored_movie_checks(INT16_PATH, result, integer=True)
    roi_error, roi_centres = roi_mean_error(INT16_PATH, result)
    float32_states = stored_qc_states(FLOAT32_PATH, result)
    int16_states = stored_qc_states(INT16_PATH, result, integer=True)
    qc_disagreements = int(np.sum(float32_states != int16_states))

    float32_size = FLOAT32_PATH.stat().st_size
    int16_size = INT16_PATH.stat().st_size
    size_ratio = int16_size / float32_size
    clipped = signal_clipped or control_clipped
    maximum_rounding_error = max(signal_rounding_error, control_rounding_error)
    int16_passed = (
        not validation_errors['int16']
        and size_ratio <= 0.75
        and not clipped
        and maximum_rounding_error <= 0.5
        and int16_checks['maximum_error_counts'] <= 0.5
        and int16_checks['valid_bounds_exact']
        and int16_checks['invalid_pixels_are_zero']
        and roi_error <= 0.5
        and qc_disagreements == 0
        )
    report = {
        'frame_window': [FRAME_START, FRAME_START + N_FRAMES - 1],
        'session_start_time': session_start_time.isoformat(),
        'session_time_source': session_time_source,
        'focal_loss_frames': int(np.sum(
            result['quality']['recommended_state'] == 'focal_loss')),
        'focal_loss_episodes': result['quality']['focal_loss_episodes'],
        'channel_offset': result['channel_offset'],
        'reference_fallback': bool(result['reference_info']['reference_fallback']),
        'validation_errors': validation_errors,
        'metadata_read_seconds': {
            name: values[0] for name, values in metadata_times.items()
            },
        'metadata_values': {
            name: {
                'session_start_time': values[1],
                'sampling_frequency_hz': values[2],
                }
            for name, values in metadata_times.items()
            },
        'float32_checks': float32_checks,
        'int16_checks': int16_checks,
        'conversion_maximum_error_counts': maximum_rounding_error,
        'clipped_pixels': clipped,
        'roi_centres_yx': roi_centres,
        'roi_maximum_mean_error_counts': roi_error,
        'qc_state_disagreements': qc_disagreements,
        'float32_size_bytes': float32_size,
        'int16_size_bytes': int16_size,
        'int16_to_float32_size_ratio': size_ratio,
        'recommended_dtype': 'int16' if int16_passed else 'float32',
        }

    if validation_errors['float32'] or validation_errors['int16']:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    if not float32_checks['float32_bitwise_exact']:
        raise AssertionError('float32 movie did not round-trip exactly')
    if max(value[0] for value in metadata_times.values()) >= 1:
        raise AssertionError('metadata-only opening exceeded one second')
    print(json.dumps(report, indent=2, default=lambda value: value.tolist()))
    return report


if __name__ == '__main__':
    run_storage_proof()

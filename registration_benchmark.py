'''
Created on 14 August 2026

make and measure the motion-correction benchmark

@author: Dinghao Luo
'''

#%% imports
from argparse import ArgumentParser
from csv import DictWriter
from pathlib import Path
import importlib.metadata
import subprocess
import sys
import time

import numpy as np
import tifffile


#%% paths
PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_ROOT = PROJECT_ROOT / 'examples'
BENCHMARK_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark'


#%% synthetic planes
def _normalise(image):
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99.7])
    return np.clip((image - low) / (high - low), 0, 1)


def make_planes(control, below, above):
    from scipy.ndimage import gaussian_filter

    control = _normalise(control)
    below = _normalise(below)
    above = _normalise(above)

    # the outer planes are deliberately severe stress cases: a single 2D
    # reference cannot tell us which structures should enter from another z plane
    planes = np.stack([
        0.25 * gaussian_filter(control, 2.5) + 0.75 * gaussian_filter(below, 0.6),
        0.72 * gaussian_filter(control, 1.1) + 0.28 * gaussian_filter(below, 0.8),
        control,
        0.70 * gaussian_filter(control, 1.3) + 0.30 * gaussian_filter(above, 0.7),
        0.22 * gaussian_filter(control, 2.8) + 0.78 * gaussian_filter(above, 0.5),
        ]).astype(np.float32)
    return np.clip(planes, 0, 1)


def plane_at(planes, z):
    z = float(np.clip(z, -2, 2)) + 2
    lower = int(np.floor(z))
    upper = min(lower + 1, 4)
    fraction = z - lower
    return (1 - fraction) * planes[lower] + fraction * planes[upper]


#%% known motion
def _soft_pulse(n_frames, start, length):
    pulse = np.zeros(n_frames, dtype=np.float32)
    if start >= n_frames:
        return pulse
    stop = min(start + length, n_frames)
    phase = np.linspace(0, np.pi, stop - start, dtype=np.float32)
    pulse[start:stop] = np.sin(phase) ** 2
    return pulse


def motion_truth(n_frames, shape, seed=42):
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    time = np.arange(n_frames, dtype=np.float32)

    drift_y = gaussian_filter1d(np.cumsum(rng.normal(0, 0.018, n_frames)), 20)
    drift_x = gaussian_filter1d(np.cumsum(rng.normal(0, 0.018, n_frames)), 20)
    shift_y = drift_y - np.median(drift_y)
    shift_x = drift_x - np.median(drift_x)

    # 15 August 2026: supplied bouts centre on 11 frames, with a 19-24 frame upper quartile
    start = 20
    while start < n_frames:
        length = int(rng.integers(15, 31))
        pulse = _soft_pulse(n_frames, start, length)
        phase = np.linspace(-1, 1, length, dtype=np.float32)[:n_frames - start]
        angle = rng.uniform(0, 2 * np.pi) + rng.uniform(-0.6, 0.6) * phase
        amplitude = rng.uniform(1.5, 5.0)
        if rng.random() < 0.15:
            amplitude *= 1.4
        shift_y[start:start + len(phase)] += amplitude * pulse[start:start + len(phase)] * np.sin(angle)
        shift_x[start:start + len(phase)] += amplitude * pulse[start:start + len(phase)] * np.cos(angle)
        start += int(rng.integers(45, 91))

    height, width = shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    y = 2 * y / (height - 1) - 1
    x = 2 * x / (width - 1) - 1
    basis_y = np.stack([
        np.sin(np.pi * x) * np.cos(np.pi * y / 2),
        x * y,
        ]).astype(np.float32)
    basis_x = np.stack([
        np.cos(np.pi * y) * np.cos(np.pi * x / 2),
        0.5 * (x * x - y * y),
        ]).astype(np.float32)
    basis_y -= basis_y.mean(axis=(1, 2), keepdims=True)
    basis_x -= basis_x.mean(axis=(1, 2), keepdims=True)

    block = np.arange(n_frames) % 1000
    coefficient = np.zeros((n_frames, 2), dtype=np.float32)
    for block_start in range(0, n_frames, 1000):
        for centre in range(block_start + 515, min(block_start + 750, n_frames), 35):
            length = int(rng.integers(15, 31))
            start = centre - length // 2
            pulse = _soft_pulse(n_frames, start, length)
            angle = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(1.3, 2.1)
            coefficient[:, 0] += amplitude * pulse * np.sin(angle)
            coefficient[:, 1] += amplitude * pulse * np.cos(angle)
    nonrigid = np.linalg.norm(coefficient, axis=1) > 0.05

    z = 0.12 * np.sin(time / 410)
    for start, length, level in [
            (310, 12, 1.0),
            (810, 15, -2.0),
            (1310, 15, -1.0),
            (1810, 15, 2.0),
            ]:
        pulse = _soft_pulse(n_frames, start, length)
        z += pulse * (level - z)

    ambiguity = np.zeros(n_frames, dtype=np.float32)
    for start in (455, 955, 1455, 1955):
        ambiguity = np.maximum(ambiguity, _soft_pulse(n_frames, start, 15))
    ambiguous = ambiguity > 0.05
    estimable = (~ambiguous) & (np.abs(z) < 1.5)

    scenario = np.full(n_frames, 'intensity', dtype='<U12')
    scenario[block < 250] = 'clean'
    scenario[nonrigid] = 'nonrigid'
    scenario[(np.abs(z) >= 0.5)] = 'focal'
    scenario[ambiguous] = 'ambiguous'

    return {
        'shift_y': shift_y.astype(np.float32),
        'shift_x': shift_x.astype(np.float32),
        'basis_y': basis_y,
        'basis_x': basis_x,
        'coefficient': coefficient,
        'z': z.astype(np.float32),
        'ambiguity': ambiguity,
        'ambiguous': ambiguous,
        'estimable': estimable,
        'scenario': scenario,
        }


def displacement_at(truth, frame, y, x):
    frame = np.atleast_1d(frame)
    basis_y = truth['basis_y'][:, y, x]
    basis_x = truth['basis_x'][:, y, x]
    local_y = truth['coefficient'][frame] @ basis_y
    local_x = truth['coefficient'][frame] @ basis_x
    return (
        truth['shift_y'][frame, None] + local_y,
        truth['shift_x'][frame, None] + local_x,
        )


def _field_limits(truth, shape):
    n_frames = len(truth['shift_y'])
    height, width = shape
    bounds = np.empty((n_frames, 4), dtype=np.int16)
    valid_mask_bits = np.empty((n_frames, (height * width + 7) // 8), dtype=np.uint8)
    jacobian_min = np.empty(n_frames, dtype=np.float32)
    jacobian_max = np.empty(n_frames, dtype=np.float32)

    ypix, xpix = np.mgrid[:height, :width].astype(np.float32)
    y = 2 * ypix / (height - 1) - 1
    x = 2 * xpix / (width - 1) - 1
    dy = 2 / (height - 1)
    dx = 2 / (width - 1)
    derivatives = [
        (
            -np.pi / 2 * np.sin(np.pi * x) * np.sin(np.pi * y / 2) * dy,
            np.pi * np.cos(np.pi * x) * np.cos(np.pi * y / 2) * dx,
            -np.pi * np.sin(np.pi * y) * np.cos(np.pi * x / 2) * dy,
            -np.pi / 2 * np.cos(np.pi * y) * np.sin(np.pi * x / 2) * dx,
            ),
        (x * dy, y * dx, -y * dy, x * dx),
        ]

    for frame in range(n_frames):
        field_y = truth['shift_y'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_y'], axis=0)
        field_x = truth['shift_x'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_x'], axis=0)
        bounds[frame] = (
            np.ceil(max(0, field_y.max())),
            height + np.floor(min(0, field_y.min())),
            np.ceil(max(0, field_x.max())),
            width + np.floor(min(0, field_x.min())),
            )
        valid = (
            (ypix - field_y >= 0)
            & (ypix - field_y <= height - 1)
            & (xpix - field_x >= 0)
            & (xpix - field_x <= width - 1)
            )
        valid_mask_bits[frame] = np.packbits(valid.ravel())

        dy_dy = sum(c * d[0] for c, d in zip(truth['coefficient'][frame], derivatives))
        dy_dx = sum(c * d[1] for c, d in zip(truth['coefficient'][frame], derivatives))
        dx_dy = sum(c * d[2] for c, d in zip(truth['coefficient'][frame], derivatives))
        dx_dx = sum(c * d[3] for c, d in zip(truth['coefficient'][frame], derivatives))
        jacobian = (1 - dy_dy) * (1 - dx_dx) - dy_dx * dx_dy
        jacobian_min[frame] = jacobian.min()
        jacobian_max[frame] = jacobian.max()

    return bounds, valid_mask_bits, jacobian_min, jacobian_max


#%% movie generation
def _warp(image, field_y, field_x):
    from scipy.ndimage import map_coordinates

    y, x = np.mgrid[:image.shape[0], :image.shape[1]].astype(np.float32)
    background = float(np.percentile(image, 2))
    return map_coordinates(
        image,
        (y - field_y, x - field_x),
        order=1,
        mode='constant',
        cval=background,
        )


def _camera_frame(image, rng, photons, gain=1, offset=0, read_noise=10):
    photon_image = rng.poisson(np.clip(image, 0, 1) * photons)
    noise = rng.normal(0, read_noise, image.shape)
    return np.clip(gain * photon_image + offset + noise, -32768, 32767).astype(np.int16)


def make_benchmark(root=BENCHMARK_ROOT, n_frames=2000, seed=42):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    control = np.load(EXAMPLE_ROOT / 'demo_train_02_ref_mat_ch2.npy')
    below = np.load(EXAMPLE_ROOT / 'demo_train_01_ref_mat_ch2.npy')
    above = np.load(EXAMPLE_ROOT / 'demo_test_ref_mat_ch2.npy')
    planes = make_planes(control, below, above)
    truth = motion_truth(n_frames, control.shape, seed)
    bounds, valid_mask_bits, jacobian_min, jacobian_max = _field_limits(truth, control.shape)

    control_movie = tifffile.memmap(
        root / 'control.tif', shape=(n_frames, *control.shape), dtype=np.int16, bigtiff=True)
    signal_movie = tifffile.memmap(
        root / 'signal.tif', shape=(n_frames, *control.shape), dtype=np.int16, bigtiff=True)
    rng = np.random.default_rng(seed + 1)
    height, width = control.shape

    for frame in range(n_frames):
        z_plane = plane_at(planes, truth['z'][frame])
        control_plane = np.power(z_plane, 2)
        signal_plane = np.clip(
            0.30 * np.power(z_plane, 1.7) + 0.70 * np.power(z_plane, 2.2)
            * (1 + 0.14 * np.sin(frame / 29)),
            0,
            1,
            )
        field_y = truth['shift_y'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_y'], axis=0)
        field_x = truth['shift_x'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_x'], axis=0)

        control_frame = _warp(control_plane, field_y, field_x)
        signal_frame = _warp(signal_plane, field_y, field_x)
        bleaching = 1 - 0.18 * frame / (n_frames - 1)
        gain = 1 + 0.16 * np.sin(frame / 47)
        offset = 12 * np.sin(frame / 83)
        if truth['ambiguity'][frame] > 0:
            tile = control_frame[:64, :64]
            repeated_control = np.tile(tile, (height // 64, width // 64))
            repeated_signal = np.tile(signal_frame[:64, :64], (height // 64, width // 64))
            weight = truth['ambiguity'][frame]
            control_frame = (1 - weight) * control_frame + weight * repeated_control
            signal_frame = (1 - weight) * signal_frame + weight * repeated_signal

        # 14 August 2026: matched to sampled signal and control histograms in the raw TIFF
        control_movie[frame] = _camera_frame(
            bleaching * control_frame, rng, photons=65, gain=gain, offset=offset)
        signal_movie[frame] = _camera_frame(
            bleaching * signal_frame, rng, photons=90, gain=1.03 * gain, offset=offset + 5)

    control_movie.flush()
    signal_movie.flush()
    np.savez_compressed(
        root / 'truth.npz',
        **truth,
        valid_bounds=bounds,
        valid_mask_bits=valid_mask_bits,
        valid_mask_shape=np.asarray(control.shape),
        jacobian_min=jacobian_min,
        jacobian_max=jacobian_max,
        source='examples/demo_train_02_ref_mat_ch2.npy',
        sampling_frequency_hz=np.float32(30),
        seed=np.int64(seed),
        )
    reference_convergence(planes[2] ** 2, root / 'reference_convergence.csv', seed + 2)


#%% reference convergence
def _gradient_ncc(a, b):
    from scipy.ndimage import sobel

    a = np.hypot(sobel(a, axis=0), sobel(a, axis=1)).ravel()
    b = np.hypot(sobel(b, axis=0), sobel(b, axis=1)).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def reference_convergence(reference, path, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n_frames in (50, 100, 200, 500, 1000):
        mean = np.zeros_like(reference, dtype=np.float64)
        for frame in range(n_frames):
            gain = 1 + rng.uniform(-0.16, 0.16)
            offset = rng.uniform(-12, 12)
            image = _camera_frame(reference, rng, photons=65, gain=gain, offset=offset)
            image = _normalise(image)
            mean += (image - mean) / (frame + 1)
        scale = np.cov(mean.ravel(), reference.ravel(), ddof=0)[0, 1] / np.var(mean)
        adjusted = scale * (mean - mean.mean()) + reference.mean()
        rows.append({
            'frames': n_frames,
            'seconds': n_frames / 30,
            'gradient_ncc': _gradient_ncc(mean, reference),
            'adjusted_rmse': float(np.sqrt(np.mean((adjusted - reference) ** 2))),
            })

    with Path(path).open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


#%% Suite2p
def run_suite2p(root=BENCHMARK_ROOT, piecewise=False):
    from suite2p import default_ops
    from suite2p.registration import register

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    ops = default_ops()
    ops.update({
        '1Preg': False,
        'batch_size': 100,
        'block_size': [64, 64],
        'maxregshift': 0.1,
        'maxregshiftNR': 3,
        'nimg_init': 500,
        'nonrigid': piecewise,
        'smooth_sigma': 1.0,
        'smooth_sigma_time': 0,
        })
    start_time = time.perf_counter()
    n_reference = min(500, len(movie))
    reference_frames = np.asarray(
        movie[np.linspace(0, len(movie) - 1, n_reference, dtype=int)], dtype=np.int16)
    reference = register.compute_reference(reference_frames.copy(), ops=ops)
    masks = register.compute_reference_masks(reference, ops=ops)

    shift_y = np.empty(len(movie), dtype=np.float32)
    shift_x = np.empty(len(movie), dtype=np.float32)
    confidence = np.empty(len(movie), dtype=np.float32)
    local_y = []
    local_x = []
    local_confidence = []
    for start in range(0, len(movie), ops['batch_size']):
        stop = min(start + ops['batch_size'], len(movie))
        # 14 August 2026: np.asarray returned a view and Suite2p rewrote the raw TIFF
        frames = np.array(movie[start:stop], dtype=np.int16, copy=True)
        output = register.register_frames(masks, frames, ops=ops)
        shift_y[start:stop], shift_x[start:stop], confidence[start:stop] = output[1:4]
        if piecewise:
            local_y.append(output[4])
            local_x.append(output[5])
            local_confidence.append(output[6])

    result = {
        'shift_y': shift_y,
        'shift_x': shift_x,
        'confidence': confidence,
        'reference': reference,
        'seconds': np.float64(time.perf_counter() - start_time),
        'version': importlib.metadata.version('suite2p'),
        }
    if piecewise:
        yblock, xblock = masks[-1][:2]
        result.update({
            'local_y': np.concatenate(local_y),
            'local_x': np.concatenate(local_x),
            'local_confidence': np.concatenate(local_confidence),
            'tile_y': np.array(yblock).mean(axis=1),
            'tile_x': np.array(xblock).mean(axis=1),
            })
    name = 'suite2p_piecewise' if piecewise else 'suite2p_rigid'
    np.savez_compressed(root / f'{name}.npz', **result)


#%% CaImAn
def _run_caiman_piecewise_part(root, start, stop):
    from caiman.motion_correction import get_patch_centers, tile_and_correct

    root = Path(root)
    path = str(root / 'control.tif')
    movie = tifffile.memmap(path)
    with np.load(root / 'caiman_rigid.npz') as rigid:
        reference = rigid['reference']
    centres = get_patch_centers((256, 256), overlaps=(32, 32), strides=(32, 32), shifts_opencv=True)
    tile_y, tile_x = np.meshgrid(centres[0], centres[1], indexing='ij')
    local_y = np.empty((stop - start, tile_y.size), dtype=np.float32)
    local_x = np.empty((stop - start, tile_x.size), dtype=np.float32)
    reference_sum = np.zeros(reference.shape, dtype=np.float64)
    reference_count = np.zeros(reference.shape, dtype=np.int32)
    add_to_movie = -float(movie[:400].min())
    start_time = time.perf_counter()

    for result_i, frame in enumerate(movie[start:stop]):
        corrected, shifts, _, _ = tile_and_correct(
            frame,
            reference,
            strides=(32, 32),
            overlaps=(32, 32),
            max_shifts=(26, 26),
            upsample_factor_fft=10,
            max_deviation_rigid=3,
            add_to_movie=add_to_movie,
            shifts_opencv=True,
            border_nan=True,
            shifts_interpolate=True,
            )
        shifts = np.asarray(shifts)
        local_y[result_i] = -shifts[:, 0]
        local_x[result_i] = -shifts[:, 1]
        valid = np.isfinite(corrected)
        reference_sum[valid] += corrected[valid]
        reference_count[valid] += 1

    np.savez_compressed(
        root / f'caiman_piecewise_{start:04d}_{stop:04d}.npz',
        local_y=local_y,
        local_x=local_x,
        reference_sum=reference_sum,
        reference_count=reference_count,
        tile_y=tile_y.ravel(),
        tile_x=tile_x.ravel(),
        seconds=np.float64(time.perf_counter() - start_time),
        version=importlib.metadata.version('caiman'),
        )


def _run_caiman_piecewise(root):
    root = Path(root)
    n_frames = len(tifffile.memmap(root / 'control.tif'))
    parts = []
    for start in range(0, n_frames, 150):
        stop = min(start + 150, n_frames)
        subprocess.run(
            [sys.executable, str(Path(__file__)), 'caiman-part', str(start), str(stop)],
            check=True,
            )
        parts.append(root / f'caiman_piecewise_{start:04d}_{stop:04d}.npz')

    saved_parts = [np.load(path) for path in parts]
    with np.load(root / 'caiman_rigid.npz') as rigid:
        reference_seconds = float(rigid['seconds'])
    reference_sum = sum(saved['reference_sum'] for saved in saved_parts)
    reference_count = sum(saved['reference_count'] for saved in saved_parts)
    result = {
        'reference': reference_sum / np.maximum(reference_count, 1),
        'seconds': reference_seconds + sum(float(saved['seconds']) for saved in saved_parts),
        'version': saved_parts[0]['version'],
        'local_y': np.concatenate([saved['local_y'] for saved in saved_parts]),
        'local_x': np.concatenate([saved['local_x'] for saved in saved_parts]),
        'tile_y': saved_parts[0]['tile_y'],
        'tile_x': saved_parts[0]['tile_x'],
        }
    np.savez_compressed(root / 'caiman_piecewise.npz', **result)
    for saved in saved_parts:
        saved.close()


def run_caiman(root=BENCHMARK_ROOT, piecewise=False):
    if piecewise:
        # 14 August 2026: the full CaImAn process crossed the local memory limit;
        # fresh 150-frame processes run the same fixed-reference operation
        _run_caiman_piecewise(root)
        return

    from caiman.motion_correction import MotionCorrect

    root = Path(root)
    path = str(root / 'control.tif')
    n_frames = len(tifffile.memmap(path))
    splits = max(1, min(20, n_frames // 100))
    start_time = time.perf_counter()

    correction = MotionCorrect(
        [path],
        max_shifts=(26, 26),
        niter_rig=1,
        splits_rig=splits,
        strides=(32, 32),
        overlaps=(32, 32),
        splits_els=splits,
        max_deviation_rigid=3,
        shifts_opencv=True,
        border_nan=True,
        pw_rigid=False,
        shifts_interpolate=True,
        )
    correction.motion_correct(save_movie=False)
    result = {
        'reference': correction.total_template_rig,
        'seconds': np.float64(time.perf_counter() - start_time),
        'version': importlib.metadata.version('caiman'),
        }
    shifts = -np.asarray(correction.shifts_rig, dtype=np.float32)
    result.update({'shift_y': shifts[:, 0], 'shift_x': shifts[:, 1]})
    np.savez_compressed(root / 'caiman_rigid.npz', **result)


#%% FibreSight
def run_fibresight(root=BENCHMARK_ROOT):
    from fibre_sight.preprocessing import estimate_shift, make_reference

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    with np.load(root / 'truth.npz') as saved:
        truth = {name: saved[name] for name in saved.files}

    control = np.load(EXAMPLE_ROOT / 'demo_train_02_ref_mat_ch2.npy')
    below = np.load(EXAMPLE_ROOT / 'demo_train_01_ref_mat_ch2.npy')
    above = np.load(EXAMPLE_ROOT / 'demo_test_ref_mat_ch2.npy')
    known_reference = make_planes(control, below, above)[2] ** 2
    calibration = (np.arange(len(movie)) < len(movie) // 2) & truth['estimable']
    phase_whitening = np.asarray([0, 0.5, 1], dtype=np.float32)
    calibration_median = np.empty(len(phase_whitening), dtype=np.float32)
    calibration_p95 = np.empty(len(phase_whitening), dtype=np.float32)
    calibration_start = time.perf_counter()

    # 14 August 2026: phase-only correlation lost subpixel accuracy at this noise level
    for choice_i, whitening in enumerate(phase_whitening):
        displacement = []
        for frame in movie[:len(movie) // 2]:
            estimate = estimate_shift(
                known_reference, frame, whitening=float(whitening), check_tiles=False)
            displacement.append([-estimate['shift_y'], -estimate['shift_x']])
        displacement = np.asarray(displacement)
        error, _, _, valid = registration_errors(
            displacement[:, 0],
            displacement[:, 1],
            truth['shift_y'][:len(displacement)],
            truth['shift_x'][:len(displacement)],
            calibration[:len(displacement)],
            truth['estimable'][:len(displacement)],
            )
        calibration_median[choice_i] = np.median(error[valid])
        calibration_p95[choice_i] = np.percentile(error[valid], 95)

    whitening = float(phase_whitening[np.argmin(calibration_p95)])
    calibration_seconds = time.perf_counter() - calibration_start
    start_time = time.perf_counter()
    reference = make_reference(movie, whitening=whitening)
    shift_y = np.empty(len(movie), dtype=np.float32)
    shift_x = np.empty(len(movie), dtype=np.float32)
    peak_ratio = np.empty(len(movie), dtype=np.float32)
    tile_disagreement = np.empty(len(movie), dtype=np.float32)
    out_of_range = np.empty(len(movie), dtype=bool)
    for frame_i, frame in enumerate(movie):
        estimate = estimate_shift(reference['image'], frame, whitening=whitening)
        # truth records observed movement; the applied correction has the opposite sign
        shift_y[frame_i] = -estimate['shift_y']
        shift_x[frame_i] = -estimate['shift_x']
        peak_ratio[frame_i] = estimate['peak_ratio']
        tile_disagreement[frame_i] = estimate['tile_disagreement']
        out_of_range[frame_i] = estimate['out_of_range']

    np.savez_compressed(
        root / 'fibresight_rigid.npz',
        shift_y=shift_y,
        shift_x=shift_x,
        peak_ratio=peak_ratio,
        tile_disagreement=tile_disagreement,
        out_of_range=out_of_range,
        reference=reference['image'],
        reference_accepted=reference['accepted'],
        phase_whitening=phase_whitening,
        calibration_median=calibration_median,
        calibration_p95=calibration_p95,
        selected_whitening=np.float32(whitening),
        calibration_seconds=np.float64(calibration_seconds),
        seconds=np.float64(time.perf_counter() - start_time),
        version=importlib.metadata.version('fibre-sight'),
        )


#%% review movie
def make_review_movie(root=BENCHMARK_ROOT):
    import av
    import cv2
    from fibre_sight.preprocessing import warp_frame

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    with np.load(root / 'truth.npz') as saved:
        truth = {name: saved[name] for name in saved.files}

    names = ['suite2p_rigid', 'caiman_rigid', 'fibresight_rigid']
    results = []
    calibration = (np.arange(len(movie)) < len(movie) // 2) & truth['estimable']
    known = np.column_stack([truth['shift_y'], truth['shift_x']])
    for name in names:
        with np.load(root / f'{name}.npz') as saved:
            movement = np.column_stack([saved['shift_y'], saved['shift_x']])
        offset = np.median(movement[calibration] - known[calibration], axis=0)
        results.append((movement, offset))

    sample = np.asarray(movie[np.linspace(0, len(movie) - 1, 100, dtype=int)])
    low, high = np.percentile(sample, [0.1, 99.9])
    height, width = movie.shape[1:]
    path = EXAMPLE_ROOT / 'rigid_registration_benchmark.mp4'
    video = av.open(path, 'w')
    stream = video.add_stream(
        'libx264', rate=round(float(truth['sampling_frequency_hz'])),
        options={'crf': '23', 'preset': 'medium'},
        )
    shown_height = 3 * height // 4
    shown_width = 3 * width // 4
    margin = 8
    gutter = 16
    title_height = 28
    stream.width = 2 * margin + 4 * shown_width + 3 * gutter
    stream.height = shown_height + 2 * margin + title_height
    stream.pix_fmt = 'yuv420p'

    labels = ['raw', 'Suite2p rigid', 'CaImAn rigid', 'FibreSight rigid']
    sections = [
        ('ordinary movement', 0, 300),
        ('non-rigid movement', 500, 800),
        ('focal-plane stress', 760, 1060),
        ]
    for section, start, stop in sections:
        for frame_i in range(start, stop):
            frame = movie[frame_i]
            panels = [np.asarray(frame, dtype=np.float32)]
            for movement, offset in results:
                correction = -movement[frame_i] + offset
                registered, _ = warp_frame(frame, *correction)
                panels.append(registered)

            shown = []
            for panel, label in zip(panels, labels):
                panel = np.nan_to_num((panel - low) / (high - low), nan=0)
                panel = np.clip(255 * panel, 0, 255).astype(np.uint8)
                panel = cv2.resize(panel, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
                cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                shown.append(panel)
            canvas = np.full((stream.height, stream.width, 3), 24, dtype=np.uint8)
            cv2.putText(
                canvas, section, (margin, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1,
                )
            for panel_i, panel in enumerate(shown):
                x = margin + panel_i * (shown_width + gutter)
                y = margin + title_height
                canvas[y:y + shown_height, x:x + shown_width] = panel
            video_frame = av.VideoFrame.from_ndarray(canvas, format='bgr24')
            for packet in stream.encode(video_frame):
                video.mux(packet)
    for packet in stream.encode():
        video.mux(packet)
    video.close()
    return path


#%% errors
def registration_errors(estimate_y, estimate_x, truth_y, truth_x, calibration, estimable):
    offset_y = np.median(estimate_y[calibration] - truth_y[calibration])
    offset_x = np.median(estimate_x[calibration] - truth_x[calibration])
    error = np.hypot(
        estimate_y - offset_y - truth_y,
        estimate_x - offset_x - truth_x,
        )
    return error, float(offset_y), float(offset_x), estimable & np.isfinite(error)


def _metric_rows(name, result, truth):
    from scipy.interpolate import RegularGridInterpolator

    n_frames = len(truth['shift_y'])
    calibration_frames = np.arange(n_frames) < n_frames // 2
    calibration = calibration_frames & truth['estimable']
    height, width = truth['basis_y'].shape[1:]
    # 14 August 2026: one grid keeps each competitor's tile layout out of the score
    sample_y = np.linspace(32, height - 33, 7).round().astype(int)
    sample_x = np.linspace(32, width - 33, 7).round().astype(int)
    grid_y, grid_x = np.meshgrid(sample_y, sample_x, indexing='ij')
    points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    truth_y, truth_x = displacement_at(
        truth, np.arange(n_frames), grid_y.ravel(), grid_x.ravel())

    if 'local_y' in result:
        estimate_y = result['local_y']
        estimate_x = result['local_x']
        if 'shift_y' in result:
            estimate_y = estimate_y + result['shift_y'][:, None]
            estimate_x = estimate_x + result['shift_x'][:, None]

        tile_y = np.unique(result['tile_y'])
        tile_x = np.unique(result['tile_x'])
        field_y = np.empty((len(tile_y), len(tile_x), n_frames), dtype=np.float32)
        field_x = np.empty_like(field_y)
        for tile, (y, x) in enumerate(zip(result['tile_y'], result['tile_x'])):
            iy = np.searchsorted(tile_y, y)
            ix = np.searchsorted(tile_x, x)
            field_y[iy, ix] = estimate_y[:, tile]
            field_x[iy, ix] = estimate_x[:, tile]
        estimate_y = RegularGridInterpolator((tile_y, tile_x), field_y)(points).T
        estimate_x = RegularGridInterpolator((tile_y, tile_x), field_x)(points).T
    else:
        estimate_y = np.broadcast_to(result['shift_y'][:, None], truth_y.shape)
        estimate_x = np.broadcast_to(result['shift_x'][:, None], truth_x.shape)

    calibration = np.broadcast_to(calibration[:, None], estimate_y.shape)
    estimable = np.broadcast_to(truth['estimable'][:, None], estimate_y.shape)
    scenario = np.broadcast_to(truth['scenario'][:, None], estimate_y.shape)
    half = np.broadcast_to(calibration_frames[:, None], estimate_y.shape)

    error, offset_y, offset_x, valid = registration_errors(
        estimate_y, estimate_x, truth_y, truth_x, calibration, estimable)
    rows = []
    groups = [('all', np.ones_like(valid)), ('calibration', half), ('heldout', ~half)]
    groups += [(item, scenario == item) for item in np.unique(scenario)]
    for group, selected in groups:
        values = error[valid & selected]
        if len(values) == 0:
            continue
        rows.append({
            'method': name,
            'group': group,
            'n': len(values),
            'median_error_px': np.median(values),
            'p95_error_px': np.percentile(values, 95),
            'over_1px_fraction': np.mean(values > 1),
            'offset_y_px': offset_y,
            'offset_x_px': offset_x,
            'seconds': float(result['seconds']),
            })
    return rows


def measure(root=BENCHMARK_ROOT):
    root = Path(root)
    with np.load(root / 'truth.npz') as saved:
        truth = {name: saved[name] for name in saved.files}
    rows = []
    for name in (
            'suite2p_rigid', 'suite2p_piecewise',
            'caiman_rigid', 'caiman_piecewise', 'fibresight_rigid',
            ):
        with np.load(root / f'{name}.npz') as saved:
            result = {field: saved[field] for field in saved.files}
        rows.extend(_metric_rows(name, result, truth))

    with (root / 'metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


#%% command line
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'step',
        choices=['make', 'suite2p', 'caiman', 'caiman-part', 'fibresight', 'movie', 'measure'],
        )
    parser.add_argument('start', nargs='?', type=int)
    parser.add_argument('stop', nargs='?', type=int)
    parser.add_argument('--piecewise', action='store_true')
    parser.add_argument('--frames', type=int, default=2000)
    args = parser.parse_args()

    if args.step == 'make':
        make_benchmark(n_frames=args.frames)
    elif args.step == 'suite2p':
        run_suite2p(piecewise=args.piecewise)
    elif args.step == 'caiman':
        run_caiman(piecewise=args.piecewise)
    elif args.step == 'caiman-part':
        _run_caiman_piecewise_part(BENCHMARK_ROOT, args.start, args.stop)
    elif args.step == 'fibresight':
        run_fibresight()
    elif args.step == 'movie':
        make_review_movie()
    else:
        measure()


if __name__ == '__main__':
    main()

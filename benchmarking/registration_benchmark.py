'''
Created on 14 August 2026
Modified on 16 August 2026
Modified on 17 August 2026 to move the benchmark out of the repository root

make and measure the motion-correction benchmark

@author: Dinghao Luo
'''

#%% imports
from concurrent.futures import ThreadPoolExecutor
from csv import DictWriter
from multiprocessing import get_context
from pathlib import Path
import importlib.metadata
import os
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np
import tifffile


#%% paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / 'examples'
BENCHMARK_ROOT = PROJECT_ROOT / 'workspace' / 'registration-benchmark'
BENCHMARK_FONT_ROOT = (
    PROJECT_ROOT / 'src' / 'fibre_sight' / 'assets' / 'fonts' / 'mononoki')
EXAMPLE_REFERENCES = (
    'demo_train_01_ref_mat_ch2.npy',
    'demo_train_02_ref_mat_ch2.npy',
    'demo_test_ref_mat_ch2.npy',
    )
MOTION_RECIPES = ('ordinary_motion', 'large_motion', 'local_deformation', 'focal_change')
# 15 August 2026: 50 and 200 expose early convergence; scored runs use 500 or 1000
REFERENCE_FRAME_COUNTS = (50, 200, 500, 1000)
PYFLOWREG_COMMIT = '126d1996c24b330bec20e7268937f9122fd2f4ab'
PATCHWARP_ROOT = PROJECT_ROOT / 'workspace' / 'dev' / 'sources' / 'PatchWarp'
PATCHWARP_COMMIT = '7cac6307b6d3aa107baecd86d8085823b437fbb1'
BENCHMARK_MATLAB_ROOT = PROJECT_ROOT / 'benchmarking' / 'matlab'
MATLAB_PATH = Path(os.environ.get(
    'MATLAB_PATH', '/Applications/MATLAB_R2022b.app/bin/matlab'))
_BENCHMARK_FONT_READY = False


def _format_benchmark_plots():
    global _BENCHMARK_FONT_READY
    import matplotlib
    from matplotlib import font_manager

    if not _BENCHMARK_FONT_READY:
        for font_path in sorted(BENCHMARK_FONT_ROOT.glob('*.ttf')):
            font_manager.fontManager.addfont(str(font_path))
        _BENCHMARK_FONT_READY = True
    matplotlib.rcParams.update({
        'font.family': 'mononoki',
        'font.monospace': ['mononoki'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        })


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

    # 15 August 2026: unrelated saved references make the severe correspondence-loss cases
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


def _focus_episode(n_frames, start, length):
    pulse = np.zeros(n_frames, dtype=np.float32)
    ramp = 3
    edge = np.sin(np.linspace(0, np.pi / 2, ramp + 2)[1:-1]) ** 2
    episode = np.ones(length, dtype=np.float32)
    episode[:ramp] = edge
    episode[-ramp:] = edge[::-1]
    pulse[start:start + length] = episode
    return pulse


def motion_truth(n_frames, shape, seed=42, recipe='ordinary_motion'):
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    time = np.arange(n_frames, dtype=np.float32)

    if recipe == 'ordinary_motion':
        drift_sd = 0.018
        bout_gap = (45, 91)
        bout_length = (15, 31)
        bout_amplitude = (1.5, 5.0)
    elif recipe == 'large_motion':
        drift_sd = 0.022
        bout_gap = (45, 91)
        bout_length = (15, 31)
        bout_amplitude = (2.0, 6.0)
    elif recipe == 'local_deformation':
        drift_sd = 0.018
        bout_gap = (45, 91)
        bout_length = (15, 31)
        bout_amplitude = (1.5, 4.5)
    elif recipe == 'focal_change':
        drift_sd = 0.018
        bout_gap = (45, 91)
        bout_length = (15, 31)
        bout_amplitude = (1.5, 4.5)
    elif recipe == 'optical_defocus':
        drift_sd = 0.018
        bout_gap = (45, 91)
        bout_length = (15, 31)
        bout_amplitude = (1.5, 4.5)
    else:
        raise ValueError(f'unknown motion recipe: {recipe}')

    drift_y = gaussian_filter1d(np.cumsum(rng.normal(0, drift_sd, n_frames)), 20)
    drift_x = gaussian_filter1d(np.cumsum(rng.normal(0, drift_sd, n_frames)), 20)
    shift_y = drift_y - np.median(drift_y)
    shift_x = drift_x - np.median(drift_x)

    # 15 August 2026: the supplied lateral bouts lasted about 0.5-1.0 s
    half = n_frames // 2
    for half_start, half_stop in ((0, half), (half, n_frames)):
        start = half_start + 20
        while start + bout_length[0] <= half_stop:
            length = int(rng.integers(
                bout_length[0], min(bout_length[1], half_stop - start + 1)))
            pulse = _soft_pulse(n_frames, start, length)
            phase = np.linspace(-1, 1, length, dtype=np.float32)
            angle = rng.uniform(0, 2 * np.pi) + rng.uniform(-0.6, 0.6) * phase
            amplitude = rng.uniform(*bout_amplitude)
            shift_y[start:start + length] += amplitude * pulse[start:start + length] * np.sin(angle)
            shift_x[start:start + length] += amplitude * pulse[start:start + length] * np.cos(angle)
            start += int(rng.integers(*bout_gap))

    if recipe == 'large_motion':
        direction = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(6, 8)
        shift_y += np.linspace(0, distance * np.sin(direction), n_frames)
        shift_x += np.linspace(0, distance * np.cos(direction), n_frames)
        for centre in (int(0.36 * n_frames), int(0.86 * n_frames)):
            length = int(rng.integers(24, 31))
            pulse = _soft_pulse(n_frames, centre - length // 2, length)
            angle = rng.uniform(0, 2 * np.pi)
            amplitude = rng.uniform(10, 12)
            shift_y += amplitude * pulse * np.sin(angle)
            shift_x += amplitude * pulse * np.cos(angle)

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

    coefficient = np.zeros((n_frames, 2), dtype=np.float32)
    if recipe in ('local_deformation', 'focal_change'):
        for half_start, half_stop in ((0, half), (half, n_frames)):
            if recipe == 'local_deformation':
                first = half_start + int(0.15 * (half_stop - half_start))
                stop = half_start + int(0.45 * (half_stop - half_start))
                local_gap = 27
                local_amplitude = (1.5, 2.3)
            else:
                first = half_start + int(0.43 * (half_stop - half_start))
                stop = half_start + int(0.60 * (half_stop - half_start))
                local_gap = 35
                local_amplitude = (1.2, 1.8)
            for centre in range(first, stop, local_gap):
                length = int(rng.integers(18, 31))
                start = centre - length // 2
                pulse = _soft_pulse(n_frames, start, length)
                angle = rng.uniform(0, 2 * np.pi)
                amplitude = rng.uniform(*local_amplitude)
                coefficient[:, 0] += amplitude * pulse * np.sin(angle)
                coefficient[:, 1] += amplitude * pulse * np.cos(angle)
    nonrigid = np.linalg.norm(coefficient, axis=1) > 0.05

    z = np.zeros(n_frames, dtype=np.float32)
    focal = np.zeros(n_frames, dtype=bool)
    defocus_sigma_px = np.zeros(n_frames, dtype=np.float32)
    contrast_loss_fraction = np.zeros(n_frames, dtype=np.float32)
    blank_frame = np.zeros(n_frames, dtype=bool)
    saturated_frame = np.zeros(n_frames, dtype=bool)
    ambiguity = np.zeros(n_frames, dtype=np.float32)
    if recipe == 'focal_change':
        z = 0.35 * np.sin(2 * np.pi * time / max(n_frames * 0.82, 1))
        # 15 August 2026: 0.25-0.50 s includes the softened entry and exit
        positions = (0.18, 0.70, 0.68, 0.20)
        levels = (1, -2, -1, 2)
        for half_i, (half_start, half_stop) in enumerate(((0, half), (half, n_frames))):
            for episode_i in range(2):
                position = positions[2 * half_i + episode_i]
                start = half_start + int(position * (half_stop - half_start))
                length = int(rng.integers(8, 16))
                pulse = _focus_episode(n_frames, start, length)
                z += pulse * (levels[2 * half_i + episode_i] - z)
                focal |= pulse > 0
            start = half_start + int(0.90 * (half_stop - half_start))
            ambiguity = np.maximum(ambiguity, _soft_pulse(n_frames, start, 15))
    elif recipe == 'optical_defocus':
        severities = ((2, 0.10), (4, 0.17), (6, 0.23), (8, 0.30))
        positions = (0.14, 0.34, 0.57, 0.78)
        lengths = (9, 12, 15, 15)
        for half_i, (half_start, half_stop) in enumerate(((0, half), (half, n_frames))):
            order = severities if half_i == 0 else severities[::-1]
            for position, length, (peak_sigma, peak_loss) in zip(
                    positions, lengths, order):
                start = half_start + int(position * (half_stop - half_start))
                pulse = _focus_episode(n_frames, start, length)
                episode = pulse > 0
                # 17 August 2026: every labelled edge has at least 2 px blur and 10% loss;
                # otherwise onset accuracy would partly measure our softened pulse definition
                defocus_sigma_px[episode] = 2 + pulse[episode] * (peak_sigma - 2)
                contrast_loss_fraction[episode] = 0.10 + pulse[episode] * (peak_loss - 0.10)
                focal |= episode
            for position in (0.06, 0.48):
                blank_frame[half_start + int(position * (half_stop - half_start))] = True
            for position in (0.08, 0.50):
                saturated_frame[half_start + int(position * (half_stop - half_start))] = True
    ambiguous = ambiguity > 0.05
    detector_artifact = blank_frame | saturated_frame
    estimable = (~ambiguous) & (np.abs(z) < 1.5) & ~detector_artifact

    base_scenario = (
        'z_drift' if recipe == 'focal_change'
        else 'defocus' if recipe == 'optical_defocus'
        else 'rigid'
        )
    scenario = np.full(n_frames, base_scenario, dtype='<U12')
    scenario[nonrigid] = 'nonrigid'
    scenario[focal] = 'focal'
    scenario[ambiguous] = 'ambiguous'
    scenario[detector_artifact] = 'artifact'

    return {
        'shift_y': shift_y.astype(np.float32),
        'shift_x': shift_x.astype(np.float32),
        'basis_y': basis_y,
        'basis_x': basis_x,
        'coefficient': coefficient,
        'z': z.astype(np.float32),
        'defocus_sigma_px': defocus_sigma_px,
        'contrast_loss_fraction': contrast_loss_fraction,
        'blank_frame': blank_frame,
        'saturated_frame': saturated_frame,
        'nonrigid': nonrigid,
        'focal': focal,
        'ambiguity': ambiguity,
        'ambiguous': ambiguous,
        'estimable': estimable,
        'scenario': scenario,
        'recipe': np.asarray(recipe),
        }


def displacement_at(truth, frame, y, x):
    from scipy.ndimage import map_coordinates

    frame = np.atleast_1d(frame)
    target_y = np.broadcast_to(y, (len(frame), len(y))).astype(np.float64).copy()
    target_x = np.broadcast_to(x, target_y.shape).astype(np.float64).copy()
    source_y = target_y.copy()
    source_x = target_x.copy()

    # 16 August 2026: competitor fields sample the moved image from reference-grid positions
    # six fixed-point steps leave less than 1e-6 px error for the synthetic Jacobian range
    for _ in range(6):
        coordinates = np.stack([source_y.ravel(), source_x.ravel()])
        basis_y = np.stack([
            map_coordinates(basis, coordinates, order=1).reshape(source_y.shape)
            for basis in truth['basis_y']
            ])
        basis_x = np.stack([
            map_coordinates(basis, coordinates, order=1).reshape(source_x.shape)
            for basis in truth['basis_x']
            ])
        local_y = np.einsum('fi,ifp->fp', truth['coefficient'][frame], basis_y)
        local_x = np.einsum('fi,ifp->fp', truth['coefficient'][frame], basis_x)
        source_y = target_y + truth['shift_y'][frame, None] + local_y
        source_x = target_x + truth['shift_x'][frame, None] + local_x
    return source_y - target_y, source_x - target_x


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


def make_benchmark(
        root=BENCHMARK_ROOT,
        n_frames=2000,
        seed=42,
        source='demo_train_02_ref_mat_ch2.npy',
        recipe='ordinary_motion',
        control_photons=65,
        bleaching=0.18,
        save_signal=True,
        ):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # 15 August 2026: write truth.npz last so an interrupted movie is incomplete
    (root / 'truth.npz').unlink(missing_ok=True)
    control = np.load(EXAMPLE_ROOT / source)
    outer_names = [name for name in EXAMPLE_REFERENCES if name != source]
    outer = [np.load(EXAMPLE_ROOT / name) for name in outer_names]
    below, above = outer
    planes = make_planes(control, below, above)
    truth_seed, control_seed, signal_seed, convergence_seed = np.random.SeedSequence(seed).spawn(4)
    truth = motion_truth(n_frames, control.shape, truth_seed, recipe)
    bounds, valid_mask_bits, jacobian_min, jacobian_max = _field_limits(truth, control.shape)
    defocus_planes = None
    defocus_plane_index = None
    if recipe == 'optical_defocus':
        from scipy.ndimage import gaussian_filter

        sigma_values, defocus_plane_index = np.unique(
            truth['defocus_sigma_px'], return_inverse=True)
        central_plane = np.power(planes[2], 2)
        # 17 August 2026: blur the latent intensity before adding photon and camera noise
        defocus_planes = np.asarray([
            gaussian_filter(central_plane, float(sigma)) if sigma else central_plane
            for sigma in sigma_values
            ], dtype=np.float32)

    description = 'FibreSight registration benchmark'
    # 15 August 2026: ImageDescription stays explicit; PatchWarp copies it to output pages
    tiff_tags = [(270, 's', len(description) + 1, description, False)]
    control_movie = tifffile.memmap(
        root / 'control.tif', shape=(n_frames, *control.shape), dtype=np.int16,
        bigtiff=True, metadata=None, extratags=tiff_tags)
    signal_movie = None
    if save_signal:
        signal_movie = tifffile.memmap(
            root / 'signal.tif', shape=(n_frames, *control.shape), dtype=np.int16,
            bigtiff=True, metadata=None, extratags=tiff_tags)
    control_rng = np.random.default_rng(control_seed)
    signal_rng = np.random.default_rng(signal_seed)
    height, width = control.shape

    for frame in range(n_frames):
        z_plane = plane_at(planes, truth['z'][frame])
        if recipe == 'optical_defocus':
            control_plane = (
                defocus_planes[defocus_plane_index[frame]]
                * (1 - truth['contrast_loss_fraction'][frame])
                )
        else:
            control_plane = np.power(z_plane, 2)
        field_y = truth['shift_y'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_y'], axis=0)
        field_x = truth['shift_x'][frame] + np.sum(
            truth['coefficient'][frame, :, None, None] * truth['basis_x'], axis=0)

        control_frame = _warp(control_plane, field_y, field_x)
        if save_signal:
            signal_plane = np.clip(
                0.30 * np.power(z_plane, 1.7) + 0.70 * np.power(z_plane, 2.2)
                * (1 + 0.14 * np.sin(frame / 29)),
                0,
                1,
                )
            if recipe == 'optical_defocus':
                sigma = truth['defocus_sigma_px'][frame]
                if sigma:
                    signal_plane = gaussian_filter(signal_plane, float(sigma))
                signal_plane *= 1 - truth['contrast_loss_fraction'][frame]
            signal_frame = _warp(signal_plane, field_y, field_x)
        brightness = 1 - bleaching * frame / (n_frames - 1)
        gain = 1 + 0.16 * np.sin(frame / 47)
        offset = 12 * np.sin(frame / 83)
        if truth['ambiguity'][frame] > 0:
            tile = control_frame[:64, :64]
            repeated_control = np.tile(tile, (height // 64, width // 64))
            weight = truth['ambiguity'][frame]
            control_frame = (1 - weight) * control_frame + weight * repeated_control
            if save_signal:
                repeated_signal = np.tile(
                    signal_frame[:64, :64], (height // 64, width // 64))
                signal_frame = (1 - weight) * signal_frame + weight * repeated_signal

        # 14 August 2026: matched to sampled signal and control histograms in the raw TIFF
        control_movie[frame] = _camera_frame(
            brightness * control_frame,
            control_rng,
            photons=control_photons,
            gain=gain,
            offset=offset,
            )
        if truth['blank_frame'][frame]:
            control_movie[frame] = 0
        elif truth['saturated_frame'][frame]:
            control_movie[frame] = np.iinfo(np.int16).max
        if save_signal:
            signal_movie[frame] = _camera_frame(
                brightness * signal_frame, signal_rng,
                photons=90, gain=1.03 * gain, offset=offset + 5)

    control_movie.flush()
    if save_signal:
        signal_movie.flush()
    np.savez_compressed(
        root / 'truth.npz',
        **truth,
        valid_bounds=bounds,
        valid_mask_bits=valid_mask_bits,
        valid_mask_shape=np.asarray(control.shape),
        jacobian_min=jacobian_min,
        jacobian_max=jacobian_max,
        latent_reference=planes[2] ** 2,
        source=f'examples/{source}',
        plane_sources=np.asarray([outer_names[0], source, outer_names[1]]),
        sampling_frequency_hz=np.float32(30),
        n_frames=np.int64(n_frames),
        seed=np.int64(seed),
        control_photons=np.float32(control_photons),
        bleaching=np.float32(bleaching),
        )
    reference_convergence(
        planes[2] ** 2,
        root / 'reference_convergence.csv',
        convergence_seed,
        photons=control_photons,
        bleaching=bleaching,
        )


#%% reference convergence
def _gradient_ncc(a, b):
    from scipy.ndimage import sobel

    a = np.hypot(sobel(a, axis=0), sobel(a, axis=1)).ravel()
    b = np.hypot(sobel(b, axis=0), sobel(b, axis=1)).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def reference_convergence(reference, path, seed, photons=65, bleaching=0):
    rng = np.random.default_rng(seed)
    rows = []
    mean = np.zeros_like(reference, dtype=np.float64)
    for frame in range(max(REFERENCE_FRAME_COUNTS)):
        gain = 1 + rng.uniform(-0.16, 0.16)
        offset = rng.uniform(-12, 12)
        brightness = 1 - bleaching * frame / (max(REFERENCE_FRAME_COUNTS) - 1)
        image = _camera_frame(
            brightness * reference, rng, photons=photons, gain=gain, offset=offset)
        image = _normalise(image)
        mean += (image - mean) / (frame + 1)
        n_frames = frame + 1
        if n_frames not in REFERENCE_FRAME_COUNTS:
            continue
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
    import cv2
    import numba
    from suite2p import default_ops
    from suite2p.registration import register
    import torch

    # 15 August 2026: this Apple GCD build ignores positive OpenCV thread limits
    cv2.setNumThreads(0)
    numba.set_num_threads(4)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    ops = default_ops()
    ops.update({
        '1Preg': False,
        'batch_size': 100,
        'block_size': [64, 64],
        'maxregshift': 0.1,
        'maxregshiftNR': 3,
        'nonrigid': piecewise,
        'smooth_sigma': 1.0,
        'smooth_sigma_time': 0,
        })
    start_time = time.perf_counter()
    n_reference = len(movie) // 2
    ops['nimg_init'] = n_reference
    reference_frames = np.array(movie[:n_reference], dtype=np.int16, copy=True)
    reference = register.compute_reference(reference_frames.copy(), ops=ops)
    masks = register.compute_reference_masks(reference, ops=ops)
    reference_seconds = time.perf_counter() - start_time

    registration_start = time.perf_counter()
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
        'reference_input_frames': np.int64(n_reference),
        'reference_seconds': np.float64(reference_seconds),
        'registration_seconds': np.float64(time.perf_counter() - registration_start),
        'version': importlib.metadata.version('suite2p'),
        }
    result['seconds'] = result['reference_seconds'] + result['registration_seconds']
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


def _run_caiman_piecewise(root):
    from caiman.motion_correction import (
        get_patch_centers,
        motion_correct_batch_pwrigid,
        motion_correct_batch_rigid,
        )

    root = Path(root)
    path = str(root / 'control.tif')
    movie = tifffile.memmap(path)
    max_shifts = tuple(round(0.1 * size) for size in movie.shape[1:])
    n_frames = len(movie)
    n_reference = n_frames // 2
    part_size = 100
    add_to_movie = -float(movie[:400].min())
    strides = (32, 32)
    overlaps = (32, 32)
    reference_parts = np.array_split(
        np.arange(n_reference), max(1, n_reference // part_size))
    parts = np.array_split(np.arange(n_frames), max(1, n_frames // part_size))

    # 15 August 2026: reset OpenCV to serial after each spawned process
    with get_context('spawn').Pool(4) as pool:
        start_time = time.perf_counter()
        _, rigid_reference, _, _ = motion_correct_batch_rigid(
            path,
            max_shifts=max_shifts,
            dview=pool,
            splits=max(1, min(10, n_reference // 100)),
            num_iter=1,
            add_to_movie=add_to_movie,
            shifts_opencv=True,
            save_movie_rigid=False,
            nonneg_movie=True,
            border_nan=True,
            subidx=slice(0, n_reference),
            shifts_interpolate=True,
            )
        # 15 August 2026: explicit frame groups stop the reference at frame 999
        _, reference, _, _, _, _, _ = motion_correct_batch_pwrigid(
            path,
            max_shifts=max_shifts,
            strides=strides,
            overlaps=overlaps,
            add_to_movie=add_to_movie,
            dview=pool,
            max_deviation_rigid=3,
            splits=reference_parts,
            num_iter=1,
            template=rigid_reference,
            shifts_opencv=True,
            save_movie=False,
            nonneg_movie=True,
            border_nan=True,
            shifts_interpolate=True,
            )
        reference_seconds = time.perf_counter() - start_time

        start_time = time.perf_counter()
        _, _, _, local_y, local_x, _, _ = motion_correct_batch_pwrigid(
            path,
            max_shifts=max_shifts,
            strides=strides,
            overlaps=overlaps,
            add_to_movie=add_to_movie,
            dview=pool,
            max_deviation_rigid=3,
            splits=parts,
            num_iter=1,
            template=reference,
            shifts_opencv=True,
            save_movie=False,
            nonneg_movie=True,
            border_nan=True,
            shifts_interpolate=True,
            )
        registration_seconds = time.perf_counter() - start_time

    centres = get_patch_centers(
        movie.shape[1:], overlaps=overlaps, strides=strides, shifts_opencv=True)
    tile_y, tile_x = np.meshgrid(*centres, indexing='ij')
    result = {
        'reference': reference,
        'reference_input_frames': np.int64(n_reference),
        'reference_seconds': np.float64(reference_seconds),
        'registration_seconds': np.float64(registration_seconds),
        'seconds': np.float64(reference_seconds + registration_seconds),
        'version': importlib.metadata.version('caiman'),
        'local_y': -np.asarray(local_y, dtype=np.float32),
        'local_x': -np.asarray(local_x, dtype=np.float32),
        'tile_y': tile_y.ravel(),
        'tile_x': tile_x.ravel(),
        }
    np.savez_compressed(root / 'caiman_piecewise.npz', **result)


def run_caiman(root=BENCHMARK_ROOT, piecewise=False):
    import cv2

    cv2.setNumThreads(0)
    if piecewise:
        # 15 August 2026: 100-frame parts kept the four-process run below the local memory limit
        _run_caiman_piecewise(root)
        return

    from caiman.motion_correction import motion_correct_batch_rigid

    root = Path(root)
    path = str(root / 'control.tif')
    movie = tifffile.memmap(path)
    max_shifts = tuple(round(0.1 * size) for size in movie.shape[1:])
    n_frames = len(movie)
    n_reference = n_frames // 2
    reference_splits = max(1, min(10, n_reference // 100))
    splits = max(1, min(20, n_frames // 100))
    add_to_movie = -float(movie[:400].min())
    start_time = time.perf_counter()
    with get_context('spawn').Pool(4) as pool:
        _, reference, _, _ = motion_correct_batch_rigid(
            path,
            max_shifts=max_shifts,
            dview=pool,
            splits=reference_splits,
            num_iter=1,
            add_to_movie=add_to_movie,
            shifts_opencv=True,
            save_movie_rigid=False,
            nonneg_movie=True,
            border_nan=True,
            subidx=slice(0, n_reference),
            shifts_interpolate=True,
            )
        reference_seconds = time.perf_counter() - start_time

        registration_start = time.perf_counter()
        _, _, _, shifts = motion_correct_batch_rigid(
            path,
            max_shifts=max_shifts,
            dview=pool,
            splits=splits,
            num_iter=1,
            template=reference,
            add_to_movie=add_to_movie,
            shifts_opencv=True,
            save_movie_rigid=False,
            nonneg_movie=True,
            border_nan=True,
            shifts_interpolate=True,
            )
        registration_seconds = time.perf_counter() - registration_start
    result = {
        'reference': reference,
        'reference_input_frames': np.int64(n_reference),
        'reference_seconds': np.float64(reference_seconds),
        'registration_seconds': np.float64(registration_seconds),
        'seconds': np.float64(reference_seconds + registration_seconds),
        'version': importlib.metadata.version('caiman'),
        }
    shifts = -np.asarray(shifts, dtype=np.float32)
    result.update({'shift_y': shifts[:, 0], 'shift_x': shifts[:, 1]})
    np.savez_compressed(root / 'caiman_rigid.npz', **result)


#%% FibreSight
def run_fibresight(root=BENCHMARK_ROOT, whitening=0, name='fibresight_rigid'):
    import cv2
    from fibre_sight import __version__
    from fibre_sight.preprocessing import (
        _estimate_shifts,
        _prepare_reference,
        make_reference,
        warp_frame,
        )

    cv2.setNumThreads(0)
    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True))
    start_time = time.perf_counter()
    n_reference = len(movie) // 2
    reference = make_reference(movie[:n_reference], whitening=whitening)
    reference_seconds = time.perf_counter() - start_time
    registration_start = time.perf_counter()
    shift_y = np.empty(len(movie), dtype=np.float32)
    shift_x = np.empty(len(movie), dtype=np.float32)
    peak_ratio = np.empty(len(movie), dtype=np.float32)
    tile_disagreement = np.empty(len(movie), dtype=np.float32)
    out_of_range = np.empty(len(movie), dtype=bool)
    search_boundary = np.empty(len(movie), dtype=bool)
    prepared = _prepare_reference(reference['image'])
    estimates = _estimate_shifts(prepared, movie, whitening=whitening)
    for frame_i, estimate in enumerate(estimates):
        # 15 August 2026: store observed movement, with the applied correction carrying the opposite sign
        shift_y[frame_i] = -estimate['shift_y']
        shift_x[frame_i] = -estimate['shift_x']
        peak_ratio[frame_i] = estimate['peak_ratio']
        tile_disagreement[frame_i] = estimate['tile_disagreement']
        out_of_range[frame_i] = estimate['out_of_range']
        search_boundary[frame_i] = estimate['search_boundary']

    def apply_shift(frame, estimate):
        return warp_frame(frame, estimate['shift_y'], estimate['shift_x'])[0]

    # 15 August 2026: include FibreSight's warps in registration time; the competitors already apply theirs
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(apply_shift, movie, estimates):
            pass

    registration_seconds = time.perf_counter() - registration_start
    np.savez_compressed(
        root / f'{name}.npz',
        shift_y=shift_y,
        shift_x=shift_x,
        peak_ratio=peak_ratio,
        tile_disagreement=tile_disagreement,
        out_of_range=out_of_range,
        search_boundary=search_boundary,
        reference=reference['image'],
        reference_aligned=reference['aligned'],
        reference_accepted=reference['accepted'],
        reference_correlation=reference['reference_correlation'],
        reference_input_frames=np.int64(n_reference),
        reference_seconds=np.float64(reference_seconds),
        registration_seconds=np.float64(registration_seconds),
        selected_whitening=np.float32(whitening),
        seconds=np.float64(reference_seconds + registration_seconds),
        version=__version__,
        git_commit=git_commit,
        git_dirty=np.bool_(git_dirty),
        )


def _pyflowreg_grid(fields, sample_y, sample_x):
    sampled = []
    for y in sample_y:
        sampled.append(np.asarray(fields[:, y, :])[:, sample_x])
    return np.concatenate(sampled, axis=1)


def run_pyflowreg(root=BENCHMARK_ROOT):
    import h5py
    from pyflowreg.motion_correction import (
        OFOptions,
        OutputFormat,
        )
    from pyflowreg.motion_correction.compensate_recording import (
        BatchMotionCorrector,
        RegistrationConfig,
        )

    root = Path(root)
    path = root / 'control.tif'
    movie = tifffile.memmap(path)
    n_frames = len(movie)
    n_reference = n_frames // 2
    sample_y = np.linspace(32, movie.shape[1] - 33, 7).round().astype(int)
    sample_x = np.linspace(32, movie.shape[2] - 33, 7).round().astype(int)

    config = RegistrationConfig(n_jobs=4, parallelization='multiprocessing')
    options = OFOptions(
        input_file=str(path),
        output_path=root / '_pyflowreg',
        output_format=OutputFormat.NULL,
        reference_frames=list(range(n_reference)),
        save_w=True,
        save_meta_info=False,
        buffer_size=400,
        quality_setting='balanced',
        alpha=4,
        weight=[1.0],
        sigma=[[1.0, 1.0, 0.1]],
        )
    reader = options.get_video_reader()
    start_time = time.perf_counter()
    reference = options.get_reference_frame(reader, registration_config=config)
    reference_seconds = time.perf_counter() - start_time

    registration_start = time.perf_counter()
    motion_corrector = BatchMotionCorrector(options, config)
    motion_corrector.run(reference)
    registration_seconds = time.perf_counter() - registration_start

    with h5py.File(options.output_path / 'w.h5', 'r') as file:
        local_x = _pyflowreg_grid(file['u'], sample_y, sample_x)
        local_y = _pyflowreg_grid(file['v'], sample_y, sample_x)
    # 15 August 2026: score the shared 7 x 7 samples; the dense field is 251 MiB
    (options.output_path / 'w.h5').unlink()
    tile_y, tile_x = np.meshgrid(sample_y, sample_x, indexing='ij')
    pyflowreg_version = importlib.metadata.version('pyflowreg')
    result = {
        'reference': np.squeeze(np.asarray(reference, dtype=np.float32)),
        'reference_input_frames': np.int64(n_reference),
        'reference_seconds': np.float64(reference_seconds),
        'registration_seconds': np.float64(registration_seconds),
        'seconds': np.float64(reference_seconds + registration_seconds),
        'version': f'pyflowreg {pyflowreg_version} @ {PYFLOWREG_COMMIT}',
        'quality_setting': 'balanced',
        'alpha': np.float32(4),
        'sigma': np.asarray([1, 1, 0.1], dtype=np.float32),
        'weight': np.float32(1),
        'workers': np.int64(4),
        'local_y': np.asarray(local_y, dtype=np.float32),
        'local_x': np.asarray(local_x, dtype=np.float32),
        'tile_y': tile_y.ravel(),
        'tile_x': tile_x.ravel(),
        }
    np.savez_compressed(root / 'pyflowreg_piecewise.npz', **result)


#%% PatchWarp
def _patchwarp_grid(
        warp_cell,
        patch_y,
        patch_x,
        nonzero_y,
        nonzero_x,
        sample_y,
        sample_x,
        edge_remove_px=0,
        ):
    kept_y = np.flatnonzero(nonzero_y) + 1
    kept_x = np.flatnonzero(nonzero_x) + 1
    grid_y, grid_x = np.meshgrid(sample_y, sample_x, indexing='ij')
    cropped_y = np.searchsorted(kept_y, grid_y.ravel() + 1) + 1
    cropped_x = np.searchsorted(
        kept_x, grid_x.ravel() + 1 - edge_remove_px) + 1

    n_frames = warp_cell.shape[2]
    local_y = np.empty((n_frames, len(cropped_y)), dtype=np.float32)
    local_x = np.empty_like(local_y)
    for point_i, (y, x) in enumerate(zip(cropped_y, cropped_x)):
        y_fields = []
        x_fields = []
        for block_y in range(warp_cell.shape[0]):
            for block_x in range(warp_cell.shape[1]):
                rows = np.ravel(patch_y[block_y, block_x])
                columns = np.ravel(patch_x[block_y, block_x])
                if y < rows[0] or y > rows[-1] or x < columns[0] or x > columns[-1]:
                    continue
                local_row = y - rows[0] + 1
                local_column = x - columns[0] + 1
                matrices = np.stack(warp_cell[block_y, block_x]).astype(np.float32)
                x_fields.append(
                    matrices[:, 0, 0] * local_column
                    + matrices[:, 0, 1] * local_row
                    + matrices[:, 0, 2] - local_column)
                y_fields.append(
                    matrices[:, 1, 0] * local_column
                    + matrices[:, 1, 1] * local_row
                    + matrices[:, 1, 2] - local_row)
        local_y[:, point_i] = np.mean(y_fields, axis=0)
        local_x[:, point_i] = np.mean(x_fields, axis=0)
    return local_y, local_x


def _write_patchwarp_tiff(movie, path):
    description = 'FibreSight registration benchmark'
    with tifffile.TiffWriter(path, bigtiff=True) as tiff:
        for frame in movie:
            # 15 August 2026: keep one plain description; PatchWarp rewrites every page
            tiff.write(
                frame,
                description=description,
                metadata=None,
                photometric='minisblack',
                )


def run_patchwarp(root=BENCHMARK_ROOT, affine=False):
    from scipy.io import loadmat

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    sample_y = np.linspace(32, movie.shape[1] - 33, 7).round().astype(int)
    sample_x = np.linspace(32, movie.shape[2] - 33, 7).round().astype(int)
    name = 'patchwarp_affine' if affine else 'patchwarp_rigid'

    with TemporaryDirectory(prefix=f'_{name}_', dir=root) as directory:
        work = Path(directory)
        source = work / 'source'
        source.mkdir()
        _write_patchwarp_tiff(movie, source / 'control.tif')

        # 16 August 2026: grid 8 leaves some 32 px focal patches without gradient samples;
        # grid 4 is PatchWarp's documented setting for moderate distortion
        warp_blocksize_attempts = (8, 4) if affine else (8,)
        for warp_blocksize in warp_blocksize_attempts:
            output = work / f'output_{warp_blocksize}'
            output.mkdir()
            matlab_code = (
                f"addpath('{BENCHMARK_MATLAB_ROOT}'); "
                f"patchwarp_benchmark('{PATCHWARP_ROOT}', "
                f"'{source / 'control.tif'}', '{output}', {str(affine).lower()}, "
                f'{warp_blocksize}, 4)')
            try:
                subprocess.run(
                    [str(MATLAB_PATH), '-singleCompThread', '-batch', matlab_code],
                    check=True,
                    )
                break
            except subprocess.CalledProcessError:
                if warp_blocksize == warp_blocksize_attempts[-1]:
                    raise

        times = loadmat(output / 'benchmark_times.mat', simplify_cells=True)
        summary = loadmat(
            output / 'pre_warp' / 'control_summary.mat', simplify_cells=True)
        correction = np.asarray(summary['t'], dtype=np.float32)
        reference = tifffile.imread(
            output / 'pre_warp' / 'target' / 'template_AVG1.tif')
        result = {
            'shift_y': -correction[:, 0],
            'shift_x': -correction[:, 1],
            'reference': reference,
            'reference_input_frames': np.int64(times['n_reference']),
            'reference_selected_frames': np.int64(np.count_nonzero(times['selected'])),
            'reference_seconds': np.float64(times['reference_seconds']),
            'registration_seconds': np.float64(
                times['rigid_seconds'] + times['affine_seconds']),
            'seconds': np.float64(
                times['reference_seconds'] + times['rigid_seconds']
                + times['affine_seconds']),
            'version': f'PatchWarp v1.3.3 @ {PATCHWARP_COMMIT}',
        }
        if affine:
            affine_result = loadmat(
                output / 'post_warp' / 'affine_transformation_matrix.mat',
                simplify_cells=True,
                )
            local_y, local_x = _patchwarp_grid(
                affine_result['warp_cell'],
                affine_result['qN_y'],
                affine_result['qN_x'],
                affine_result['nonzero_row'],
                affine_result['nonzero_column'],
                sample_y,
                sample_x,
                int(affine_result['edge_remove_pix']),
                )
            block_frames = int(affine_result['downsample_frame_num'])
            # 15 August 2026: one affine matrix is applied to each 50-frame block
            local_y = np.repeat(local_y, block_frames, axis=0)[:len(movie)]
            local_x = np.repeat(local_x, block_frames, axis=0)[:len(movie)]
            tile_y, tile_x = np.meshgrid(sample_y, sample_x, indexing='ij')
            result.update({
                'local_y': local_y,
                'local_x': local_x,
                'tile_y': tile_y.ravel(),
                'tile_x': tile_x.ravel(),
                'warp_blocksize': np.int64(times['warp_blocksize']),
                'warp_blocksize_attempts': np.asarray(
                    warp_blocksize_attempts[:warp_blocksize_attempts.index(
                        warp_blocksize) + 1],
                    dtype=np.int64,
                    ),
                })
        np.savez_compressed(root / f'{name}.npz', **result)


def run_patchwarp_references(root=BENCHMARK_ROOT):
    from scipy.io import loadmat

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    counts = [count for count in REFERENCE_FRAME_COUNTS if count <= len(movie) // 2]
    saved = {}
    rigid_path = root / 'patchwarp_rigid.npz'
    if len(movie) // 2 in counts and rigid_path.exists():
        with np.load(rigid_path) as rigid:
            saved[len(movie) // 2] = (
                rigid['reference'], float(rigid['reference_seconds']))

    fresh_counts = [count for count in counts if count not in saved]
    if fresh_counts:
        with TemporaryDirectory(prefix='_patchwarp_references_', dir=root) as directory:
            output = Path(directory)
            count_text = ' '.join(str(count) for count in fresh_counts)
            matlab_code = (
                f"addpath('{BENCHMARK_MATLAB_ROOT}'); "
                f"patchwarp_references('{PATCHWARP_ROOT}', '{root / 'control.tif'}', "
                f"'{output}', [{count_text}])")
            subprocess.run(
                [str(MATLAB_PATH), '-singleCompThread', '-batch', matlab_code], check=True)
            times = loadmat(output / 'reference_times.mat', simplify_cells=True)
            for count, seconds in zip(fresh_counts, np.atleast_1d(times['seconds'])):
                saved[count] = (
                    tifffile.imread(output / f'reference_{count}.tif'), float(seconds))
    np.savez_compressed(
        root / 'patchwarp_reference_convergence.npz',
        frame_count=np.asarray(counts),
        reference=np.asarray([saved[count][0] for count in counts]),
        seconds=np.asarray([saved[count][1] for count in counts]),
        version=f'PatchWarp v1.3.3 @ {PATCHWARP_COMMIT}',
        )


#%% reference convergence by method
def run_fibresight_references(root=BENCHMARK_ROOT):
    import cv2
    from fibre_sight import __version__
    from fibre_sight.preprocessing import make_reference

    cv2.setNumThreads(0)
    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    counts = [count for count in REFERENCE_FRAME_COUNTS if count <= len(movie) // 2]
    references = []
    seconds = []
    for count in counts:
        if count == len(movie) // 2 and (root / 'fibresight_rigid.npz').exists():
            with np.load(root / 'fibresight_rigid.npz') as saved:
                references.append(saved['reference'])
                seconds.append(float(saved['reference_seconds']))
            continue
        start_time = time.perf_counter()
        # 15 August 2026: short convergence runs keep min_frames=count // 2 from the full reference
        references.append(make_reference(
            movie[:count], min_frames=count // 2, whitening=0)['image'])
        seconds.append(time.perf_counter() - start_time)
    np.savez_compressed(
        root / 'fibresight_reference_convergence.npz',
        frame_count=np.asarray(counts),
        reference=np.asarray(references),
        seconds=np.asarray(seconds),
        version=__version__,
        )


def run_suite2p_references(root=BENCHMARK_ROOT):
    import cv2
    import numba
    from suite2p import default_ops
    from suite2p.registration import register
    import torch

    cv2.setNumThreads(0)
    numba.set_num_threads(4)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    root = Path(root)
    movie = tifffile.memmap(root / 'control.tif')
    counts = [count for count in REFERENCE_FRAME_COUNTS if count <= len(movie) // 2]
    references = []
    seconds = []
    ops = default_ops()
    ops.update({
        '1Preg': False,
        'maxregshift': 0.1,
        'smooth_sigma': 1.0,
        'smooth_sigma_time': 0,
        })
    for count in counts:
        if count == len(movie) // 2 and (root / 'suite2p_rigid.npz').exists():
            with np.load(root / 'suite2p_rigid.npz') as saved:
                references.append(saved['reference'])
                seconds.append(float(saved['reference_seconds']))
            continue
        ops['nimg_init'] = count
        start_time = time.perf_counter()
        frames = np.array(movie[:count], dtype=np.int16, copy=True)
        references.append(register.compute_reference(frames, ops=ops))
        seconds.append(time.perf_counter() - start_time)
    np.savez_compressed(
        root / 'suite2p_reference_convergence.npz',
        frame_count=np.asarray(counts),
        reference=np.asarray(references),
        seconds=np.asarray(seconds),
        version=importlib.metadata.version('suite2p'),
        )


def run_caiman_references(root=BENCHMARK_ROOT, piecewise=False):
    import cv2
    from caiman.motion_correction import (
        motion_correct_batch_pwrigid,
        motion_correct_batch_rigid,
        )

    cv2.setNumThreads(0)
    root = Path(root)
    path = str(root / 'control.tif')
    movie = tifffile.memmap(path)
    max_shifts = tuple(round(0.1 * size) for size in movie.shape[1:])
    counts = [count for count in REFERENCE_FRAME_COUNTS if count <= len(movie) // 2]
    name = 'caiman_piecewise' if piecewise else 'caiman_rigid'
    references = []
    seconds = []
    add_to_movie = -float(movie[:400].min())
    with get_context('spawn').Pool(4) as pool:
        for count in counts:
            if count == len(movie) // 2 and (root / f'{name}.npz').exists():
                with np.load(root / f'{name}.npz') as saved:
                    references.append(saved['reference'])
                    seconds.append(float(saved['reference_seconds']))
                continue
            start_time = time.perf_counter()
            _, reference, _, _ = motion_correct_batch_rigid(
                path,
                max_shifts=max_shifts,
                dview=pool,
                splits=max(1, min(10, count // 100)),
                num_iter=1,
                add_to_movie=add_to_movie,
                shifts_opencv=True,
                save_movie_rigid=False,
                nonneg_movie=True,
                border_nan=True,
                subidx=slice(0, count),
                shifts_interpolate=True,
                )
            if piecewise:
                parts = np.array_split(
                    np.arange(count), max(1, count // 100))
                _, reference, _, _, _, _, _ = motion_correct_batch_pwrigid(
                    path,
                    max_shifts=max_shifts,
                    strides=(32, 32),
                    overlaps=(32, 32),
                    add_to_movie=add_to_movie,
                    dview=pool,
                    max_deviation_rigid=3,
                    splits=parts,
                    num_iter=1,
                    template=reference,
                    shifts_opencv=True,
                    save_movie=False,
                    nonneg_movie=True,
                    border_nan=True,
                    shifts_interpolate=True,
                    )
            references.append(reference)
            seconds.append(time.perf_counter() - start_time)
    np.savez_compressed(
        root / f'{name}_reference_convergence.npz',
        frame_count=np.asarray(counts),
        reference=np.asarray(references),
        seconds=np.asarray(seconds),
        version=importlib.metadata.version('caiman'),
        )


def run_pyflowreg_references(root=BENCHMARK_ROOT):
    from pyflowreg.motion_correction import OFOptions, OutputFormat
    from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

    root = Path(root)
    path = root / 'control.tif'
    movie = tifffile.memmap(path)
    counts = [count for count in REFERENCE_FRAME_COUNTS if count <= len(movie) // 2]
    saved = {}
    result_path = root / 'pyflowreg_piecewise.npz'
    if len(movie) // 2 in counts and result_path.exists():
        with np.load(result_path) as result:
            saved[len(movie) // 2] = (
                result['reference'], float(result['reference_seconds']))

    fresh_counts = [count for count in counts if count not in saved]
    if fresh_counts:
        config = RegistrationConfig(n_jobs=4, parallelization='multiprocessing')
        options = OFOptions(
            input_file=str(path),
            output_path=root / '_pyflowreg',
            output_format=OutputFormat.NULL,
            reference_frames=[],
            save_w=True,
            save_meta_info=False,
            buffer_size=400,
            quality_setting='balanced',
            alpha=4,
            weight=[1.0],
            sigma=[[1.0, 1.0, 0.1]],
            )
        with options.get_video_reader() as reader:
            for count in fresh_counts:
                options.reference_frames = list(range(count))
                start_time = time.perf_counter()
                reference = options.get_reference_frame(
                    reader, registration_config=config)
                saved[count] = (
                    np.squeeze(np.asarray(reference, dtype=np.float32)),
                    time.perf_counter() - start_time,
                    )

    pyflowreg_version = importlib.metadata.version('pyflowreg')
    np.savez_compressed(
        root / 'pyflowreg_reference_convergence.npz',
        frame_count=np.asarray(counts),
        reference=np.asarray([saved[count][0] for count in counts]),
        seconds=np.asarray([saved[count][1] for count in counts]),
        version=f'pyflowreg {pyflowreg_version} @ {PYFLOWREG_COMMIT}',
        quality_setting='balanced',
        alpha=np.float32(4),
        sigma=np.asarray([1, 1, 0.1], dtype=np.float32),
        weight=np.float32(1),
        workers=np.int64(4),
        )


def measure_reference_convergence(root, truth):
    root = Path(root)
    latent = np.asarray(truth['latent_reference'], dtype=np.float32)
    methods = (
        ('fibresight_reference_convergence.npz', 'fibresight_rigid'),
        ('suite2p_reference_convergence.npz', 'suite2p_rigid'),
        ('caiman_rigid_reference_convergence.npz', 'caiman_rigid'),
        ('caiman_piecewise_reference_convergence.npz', 'caiman_piecewise'),
        ('patchwarp_reference_convergence.npz', 'patchwarp_rigid'),
        ('pyflowreg_reference_convergence.npz', 'pyflowreg_piecewise'),
        )
    series = {}
    aligned = []
    for path, method in methods:
        with np.load(root / path) as saved:
            series[method] = {name: saved[name] for name in saved.files}
        aligned.extend(
            _align_reference(reference, latent)[0]
            for reference in series[method]['reference']
            )

    n_reference = max(series[method]['frame_count'].max() for method in series)
    bounds = truth['valid_bounds'][:n_reference]
    motion_valid = np.zeros(latent.shape, dtype=bool)
    motion_valid[
        int(bounds[:, 0].max()):int(bounds[:, 1].min()),
        int(bounds[:, 2].max()):int(bounds[:, 3].min()),
        ] = True
    common = motion_valid & np.logical_and.reduce(
        [np.isfinite(reference) for reference in aligned])
    y, x = np.where(common)
    crop = np.s_[y.min():y.max() + 1, x.min():x.max() + 1]
    rows = []
    aligned_i = 0
    for _, method in methods:
        for count, seconds, reference in zip(
                series[method]['frame_count'], series[method]['seconds'],
                series[method]['reference']):
            fitted, _, _ = _fit_reference(aligned[aligned_i], latent, crop)
            rows.append({
                'method': method,
                'frames': int(count),
                'seconds': float(seconds),
                'gradient_ncc': _gradient_ncc(fitted[crop], latent[crop]),
                })
            aligned_i += 1

    with (root / 'method_reference_convergence.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


#%% review movie
def make_rigid_comparison_movie(
        root=BENCHMARK_ROOT,
        path=EXAMPLE_ROOT / 'rigid_registration_benchmark.mp4',
        ):
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
    sections = [('rigid benchmark', 0, min(900, len(movie)))]
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


#%% reference comparison
def _align_reference(reference, latent):
    import cv2
    from skimage.registration import phase_cross_correlation

    reference = np.asarray(reference, dtype=np.float32)
    finite = np.isfinite(reference)
    filled = np.where(finite, reference, np.nanmedian(reference))
    shift, _, _ = phase_cross_correlation(
        _normalise(latent), _normalise(filled), upsample_factor=20, normalization=None)
    matrix = np.array([[1, 0, shift[1]], [0, 1, shift[0]]], dtype=np.float32)
    aligned = cv2.warpAffine(
        reference,
        matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
        )
    return aligned, shift


def _fit_reference(reference, latent, crop):
    values = reference[crop].ravel()
    target = latent[crop].ravel()
    gain, offset = np.linalg.lstsq(
        np.column_stack([values, np.ones(len(values))]), target, rcond=None)[0]
    return gain * reference + offset, float(gain), float(offset)


def _fourier_ring_correlation(first, second):
    height, width = first.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    first_fft = np.fft.rfft2((first - first.mean()) * window)
    second_fft = np.fft.rfft2((second - second.mean()) * window)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.hypot(fy, fx)
    step = 1 / min(first.shape)
    rings = np.floor(radius / step).astype(int)

    rows = []
    for ring in range(int(0.5 / step) + 1):
        selected = rings == ring
        first_ring = first_fft[selected]
        second_ring = second_fft[selected]
        denominator = np.sqrt(
            np.sum(np.abs(first_ring) ** 2) * np.sum(np.abs(second_ring) ** 2))
        correlation = (
            np.nan if denominator == 0
            else np.real(np.sum(first_ring * second_ring.conj())) / denominator
            )
        rows.append((ring * step, float(correlation), int(selected.sum())))
    return rows


def _heldout_reference_error(reference, crop, movie, truth):
    from skimage.registration import phase_cross_correlation

    n_reference = len(movie) // 2
    ordinary = np.isin(truth['scenario'], ['rigid', 'z_drift'])
    calibration = (np.arange(len(movie)) < n_reference) & ordinary & truth['estimable']
    heldout = (np.arange(len(movie)) >= n_reference) & ordinary & truth['estimable']
    frames = np.flatnonzero(calibration | heldout)
    reference = np.asarray(reference, dtype=np.float32)
    reference = np.where(np.isfinite(reference), reference, np.nanmedian(reference))
    reference = _normalise(reference[crop])
    movement = np.empty((len(frames), 2), dtype=np.float32)
    for result_i, frame_i in enumerate(frames):
        shift, _, _ = phase_cross_correlation(
            reference,
            _normalise(np.asarray(movie[frame_i][crop])),
            upsample_factor=10,
            normalization=None,
            )
        movement[result_i] = -shift

    error, _, _, _ = registration_errors(
        movement[:, 0],
        movement[:, 1],
        truth['shift_y'][frames],
        truth['shift_x'][frames],
        calibration[frames],
        np.ones(len(frames), dtype=bool),
        )
    return error[heldout[frames]]


def measure_references(root, truth, names):
    from skimage.metrics import structural_similarity

    root = Path(root)
    latent = np.asarray(truth['latent_reference'], dtype=np.float32)
    movie = tifffile.memmap(root / 'control.tif')
    names = [name for name in names if (root / f'{name}.npz').exists()]
    results = {}
    aligned = {}
    shifts = {}
    for name in names:
        with np.load(root / f'{name}.npz') as saved:
            results[name] = {field: saved[field] for field in saved.files}
        aligned[name], shifts[name] = _align_reference(results[name]['reference'], latent)

    bounds = truth['valid_bounds'][:len(movie) // 2]
    motion_valid = np.zeros(latent.shape, dtype=bool)
    motion_valid[
        int(bounds[:, 0].max()):int(bounds[:, 1].min()),
        int(bounds[:, 2].max()):int(bounds[:, 3].min()),
        ] = True
    common = motion_valid & np.logical_and.reduce(
        [np.isfinite(aligned[name]) for name in names])
    y, x = np.where(common)
    crop = np.s_[y.min():y.max() + 1, x.min():x.max() + 1]
    latent_crop = latent[crop]
    fitted = {}
    rows = []
    frc_rows = []
    for name in names:
        fitted[name], gain, offset = _fit_reference(aligned[name], latent, crop)
        reference_crop = fitted[name][crop]
        heldout_error = _heldout_reference_error(results[name]['reference'], crop, movie, truth)
        rows.append({
            'method': name,
            'reference_input_frames': int(results[name]['reference_input_frames']),
            'reference_seconds': float(results[name]['reference_seconds']),
            'alignment_shift_y_px': float(shifts[name][0]),
            'alignment_shift_x_px': float(shifts[name][1]),
            'valid_fraction': float(np.mean(np.isfinite(aligned[name]))),
            'common_valid_fraction': float(np.mean(common)),
            'affine_gain': gain,
            'affine_offset': offset,
            'gradient_ncc': _gradient_ncc(reference_crop, latent_crop),
            'adjusted_rmse': float(np.sqrt(np.mean((reference_crop - latent_crop) ** 2))),
            'ssim': float(structural_similarity(
                reference_crop, latent_crop, data_range=np.ptp(latent_crop))),
            'heldout_frames': len(heldout_error),
            'heldout_median_error_px': float(np.median(heldout_error)),
            'heldout_p95_error_px': float(np.percentile(heldout_error, 95)),
            'heldout_over_1px_fraction': float(np.mean(heldout_error > 1)),
            })
        for frequency, correlation, n_coefficients in _fourier_ring_correlation(
                reference_crop, latent_crop):
            frc_rows.append({
                'method': name,
                'spatial_frequency_cycles_per_px': frequency,
                'frc': correlation,
                'n_coefficients': n_coefficients,
                })

    with (root / 'reference_metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'reference_frc.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=frc_rows[0])
        writer.writeheader()
        writer.writerows(frc_rows)

    import matplotlib.pyplot as plt

    _format_benchmark_plots()

    figure, axes = plt.subplots(2, len(names) + 1, figsize=(18, 6), constrained_layout=True)
    axes[0, 0].imshow(latent_crop, cmap='gray')
    axes[0, 0].set_title('latent plane')
    axes[1, 0].axis('off')
    errors = [np.abs(fitted[name][crop] - latent_crop) for name in names]
    error_high = np.percentile(np.stack(errors), 99.5)
    for column, name in enumerate(names, start=1):
        axes[0, column].imshow(fitted[name][crop], cmap='gray')
        axes[0, column].set_title(name.replace('_', ' '))
        axes[1, column].imshow(errors[column - 1], cmap='magma', vmin=0, vmax=error_high)
        axes[1, column].set_title('absolute error')
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(root / 'reference_comparison.png', dpi=180)
    plt.close(figure)


def _metric_rows(name, result, truth):
    from scipy.interpolate import RegularGridInterpolator

    n_frames = len(truth['shift_y'])
    calibration_frames = np.arange(n_frames) < n_frames // 2
    calibration = calibration_frames & truth['estimable']
    height, width = truth['basis_y'].shape[1:]
    # 14 August 2026: score every method on the same 7 x 7 grid
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
        interpolation = 'cubic' if name == 'caiman_piecewise' else 'linear'
        estimate_y = RegularGridInterpolator(
            (tile_y, tile_x), field_y, method=interpolation)(points).T
        estimate_x = RegularGridInterpolator(
            (tile_y, tile_x), field_x, method=interpolation)(points).T
    else:
        estimate_y = np.broadcast_to(result['shift_y'][:, None], truth_y.shape)
        estimate_x = np.broadcast_to(result['shift_x'][:, None], truth_x.shape)

    calibration = np.broadcast_to(calibration[:, None], estimate_y.shape)
    estimable = np.broadcast_to(truth['estimable'][:, None], estimate_y.shape)
    scenario = np.broadcast_to(truth['scenario'][:, None], estimate_y.shape)
    half = np.broadcast_to(calibration_frames[:, None], estimate_y.shape)

    error, offset_y, offset_x, _ = registration_errors(
        estimate_y, estimate_x, truth_y, truth_x, calibration, estimable)
    rows = []
    finite = np.isfinite(error)
    groups = [
        ('all', estimable),
        ('calibration', estimable & half),
        ('heldout', estimable & ~half),
        ]
    groups += [
        (item, estimable & (scenario == item)) for item in np.unique(scenario)]
    groups += [
        ('focal_stress', ~estimable & (scenario == 'focal')),
        ('ambiguous_stress', ~estimable & (scenario == 'ambiguous')),
        ]
    for group, eligible in groups:
        eligible_n = int(eligible.sum())
        if eligible_n == 0:
            continue
        values = error[eligible & finite]
        median = np.median(values) if len(values) else np.nan
        p95 = np.percentile(values, 95) if len(values) else np.nan
        over_1px = np.mean(values > 1) if len(values) else np.nan
        rows.append({
            'method': name,
            'group': group,
            'n': len(values),
            'eligible_n': eligible_n,
            'valid_fraction': len(values) / eligible_n,
            'median_error_px': median,
            'p95_error_px': p95,
            'over_1px_fraction': over_1px,
            'offset_y_px': offset_y,
            'offset_x_px': offset_x,
            'seconds': float(result['seconds']),
            })
    return rows


def measure(
        root=BENCHMARK_ROOT,
        names=(
            'suite2p_rigid', 'suite2p_piecewise',
            'caiman_rigid', 'caiman_piecewise', 'fibresight_rigid',
            ),
        compare_references=True,
        ):
    root = Path(root)
    with np.load(root / 'truth.npz') as saved:
        truth = {name: saved[name] for name in saved.files}
    rows = []
    for name in names:
        if not (root / f'{name}.npz').exists():
            # 15 August 2026: a competitor failure remains an absent score
            continue
        with np.load(root / f'{name}.npz') as saved:
            result = {field: saved[field] for field in saved.files}
        rows.extend(_metric_rows(name, result, truth))

    with (root / 'metrics.csv').open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if compare_references:
        measure_references(root, truth, names)

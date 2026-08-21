'''
Created on 14 August 2026
Modified on 18 August 2026
Modified on 19 August 2026

read paired TIFF frames, correct rigid and piecewise movement, and record registration QC

@author: Dinghao Luo
'''

#%% imports
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from functools import cache, partial
import hashlib
from itertools import groupby, islice
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter

import cv2
import h5py
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, validate
from pynwb.base import Images
from pynwb.image import GrayscaleImage, ImageSeries
from scipy import fft, ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.signal.windows import tukey
import tifffile


#%% TIFF order
_NUMBER = re.compile(r'(\d+)')
_SCANIMAGE_EPOCH = re.compile(r'^epoch\s*=\s*\[([^]]+)]', re.MULTILINE)
SEGMENTATION_REFERENCE_PERCENTILES = (1, 97)


def _sort_tiffs(paths):
    def sort_key(path):
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in _NUMBER.split(path.name)
            )

    return sorted((Path(path) for path in paths), key=sort_key)


#%% page pairing
def _multiplexed_pairs(tiff_files, signal_channel, control_channel):
    # 19 August 2026: channels are the two visible positions in each TIFF pair
    signal_offset = signal_channel - 1
    control_offset = control_channel - 1
    for path in tiff_files:
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) % 2:
                raise ValueError(f'incomplete signal/control pair: {path}')

            for page_i in range(0, len(tiff.pages), 2):
                yield (
                    tiff.pages[page_i + signal_offset],
                    tiff.pages[page_i + control_offset],
                    path,
                    path,
                    page_i + signal_offset,
                    page_i + control_offset,
                    )


def _separate_pairs(signal_tiffs, control_tiffs):
    if len(signal_tiffs) != len(control_tiffs):
        raise ValueError('signal and control TIFF lists have different lengths')

    for signal_path, control_path in zip(signal_tiffs, control_tiffs):
        with (
                tifffile.TiffFile(signal_path) as signal_tiff,
                tifffile.TiffFile(control_path) as control_tiff,
                ):
            if len(signal_tiff.pages) != len(control_tiff.pages):
                raise ValueError('paired signal and control TIFFs have different frame counts')
            for page_i in range(len(signal_tiff.pages)):
                yield (
                    signal_tiff.pages[page_i],
                    control_tiff.pages[page_i],
                    signal_path,
                    control_path,
                    page_i,
                    page_i,
                    )


#%% recording index
def index_tiffs(
        signal_tiffs,
        *,
        signal_channel,
        control_channel,
        sampling_frequency_hz,
        multiplexed=True,
        control_tiffs=None,
        signal_label=None,
        control_label=None,
        ):
    if signal_channel < 1 or control_channel < 1 or signal_channel == control_channel:
        raise ValueError('signal and control channels must be different one-based numbers')
    if sampling_frequency_hz <= 0:
        raise ValueError('sampling_frequency_hz must be positive')
    if multiplexed and {signal_channel, control_channel} != {1, 2}:
        raise ValueError('multiplexed channel numbers must be 1 and 2')

    signal_tiffs = _sort_tiffs(signal_tiffs)
    if not signal_tiffs:
        raise ValueError('no signal TIFFs found')

    if multiplexed:
        pairs = _multiplexed_pairs(signal_tiffs, signal_channel, control_channel)
        control_tiffs = signal_tiffs
    else:
        if control_tiffs is None:
            raise ValueError('control_tiffs is required for separate TIFFs')
        control_tiffs = _sort_tiffs(control_tiffs)
        pairs = _separate_pairs(signal_tiffs, control_tiffs)

    shape = None
    dtype = None
    frames = []
    # 19 August 2026: page order is the record; ScanImage counters may restart between chunks
    for frame, pair in enumerate(pairs):
        signal_page, control_page, signal_path, control_path, signal_i, control_i = pair
        if signal_page.shape != control_page.shape or signal_page.dtype != control_page.dtype:
            raise ValueError('paired signal and control frames have different shape or dtype')
        if shape is None:
            shape = signal_page.shape
            dtype = signal_page.dtype
        elif signal_page.shape != shape or signal_page.dtype != dtype:
            raise ValueError('TIFF frame shape or dtype changes within the recording')

        frames.append({
            'frame': frame,
            'signal_tiff': signal_path,
            'control_tiff': control_path,
            'signal_page': signal_i,
            'control_page': control_i,
            })

    return {
        'sampling_frequency_hz': float(sampling_frequency_hz),
        'multiplexed': bool(multiplexed),
        'signal_channel': int(signal_channel),
        'control_channel': int(control_channel),
        'signal_label': signal_label,
        'control_label': control_label,
        'shape': tuple(shape),
        'dtype': np.dtype(dtype),
        'signal_tiffs': tuple(signal_tiffs),
        'control_tiffs': tuple(control_tiffs),
        'n_frames': len(frames),
        'frames': frames,
        }


#%% reader
def read_tiffs(recording):
    path_pair = lambda frame: (frame['signal_tiff'], frame['control_tiff'])
    for paths, frames in groupby(recording['frames'], key=path_pair):
        signal_path, control_path = paths
        with ExitStack() as stack:
            signal_tiff = stack.enter_context(tifffile.TiffFile(signal_path))
            control_tiff = (
                signal_tiff if signal_path == control_path
                else stack.enter_context(tifffile.TiffFile(control_path))
                )
            for frame in frames:
                yield {
                    **frame,
                    'signal': signal_tiff.pages[frame['signal_page']].asarray(),
                    'control': control_tiff.pages[frame['control_page']].asarray(),
                    }


#%% acquisition time
def read_session_start_time(path):
    path = Path(path)
    local_timezone = datetime.now().astimezone().tzinfo
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        match = _SCANIMAGE_EPOCH.search(page.description or '')
        if match:
            values = [float(value) for value in match.group(1).split()]
            if len(values) != 6:
                raise ValueError(f'unexpected ScanImage epoch in {path}')
            seconds = int(values[5])
            start_time = datetime(
                *(int(value) for value in values[:5]), seconds,
                tzinfo=local_timezone,
                ) + timedelta(microseconds=round((values[5] - seconds) * 1e6))
            return start_time, {
                'source': 'ScanImage epoch',
                'raw_value': match.group(1),
                'timezone': str(local_timezone),
                }

        datetime_tag = page.tags.get('DateTime')
        if datetime_tag is not None:
            raw_value = str(datetime_tag.value)
            start_time = datetime.strptime(
                raw_value, '%Y:%m:%d %H:%M:%S').replace(tzinfo=local_timezone)
            return start_time, {
                'source': 'TIFF DateTime',
                'raw_value': raw_value,
                'timezone': str(local_timezone),
                }

    modified_time = path.stat().st_mtime
    if np.isfinite(modified_time):
        return datetime.fromtimestamp(modified_time, timezone.utc), {
            'source': 'file modification time',
            'raw_value': str(modified_time),
            'timezone': 'UTC',
            }

    # 19 August 2026: Cajal's birthday is unambiguously not an acquisition time
    return datetime(1852, 5, 1, tzinfo=timezone.utc), {
        'source': 'Santiago Ramon y Cajal birthday fallback',
        'raw_value': '1852-05-01',
        'timezone': 'UTC',
        }


#%% rigid registration
# 15 August 2026: full-frame and tile tapers recur for every sampled image
@cache
def _taper(shape):
    # 15 August 2026: float32 estimator copies keep the coarse FFTs compact
    # the local DFT stays complex128; a complex64 refinement moved one selected shift
    return np.outer(
        tukey(shape[0], 0.2), tukey(shape[1], 0.2)).astype(np.float32)


def _registration_image(image):
    image = np.asarray(image, dtype=np.float32)
    # 15 August 2026: centring cancels in this band-pass; positive scaling leaves its peaks unchanged
    # 15 August 2026: these match SciPy's four-sigma kernels and run faster in OpenCV
    image = cv2.GaussianBlur(
        image, (9, 9), 1, borderType=cv2.BORDER_REFLECT)
    image -= cv2.GaussianBlur(
        image, (169, 169), 21, borderType=cv2.BORDER_REFLECT)
    return image * _taper(image.shape)


def _local_dft(data, region_size, upsample, offsets):
    properties = list(zip(data.shape, [region_size] * data.ndim, offsets))
    for n_items, size, offset in properties[::-1]:
        frequency = fft.fftfreq(n_items, d=upsample)
        kernel = np.exp(
            -2j * np.pi * (np.arange(size) - offset)[:, None] * frequency[None, :]
            )
        # 15 August 2026: direct contraction avoids OpenBLAS startup for this 15-pixel search
        data = np.einsum('ij,...j->i...', kernel, data, optimize=False)
    return data


# 15 August 2026: the same two search grids serve up to 6,000 reference comparisons
@cache
def _phase_grid(shape, max_shift):
    y = np.arange(shape[0])
    x = np.arange(shape[1])
    y[y > shape[0] // 2] -= shape[0]
    x[x > shape[1] // 2] -= shape[1]
    yy, xx = np.meshgrid(y, x, indexing='ij')
    allowed = (np.abs(yy) <= max_shift[0]) & (np.abs(xx) <= max_shift[1])
    return yy, xx, allowed


def _shift_from_fft(reference_fft, frame, max_shift, upsample, whitening):
    # 15 August 2026: SciPy's FFT shortened the run with the same estimated shifts
    frame_fft = fft.fftn(frame)
    product = reference_fft * frame_fft.conj()
    if whitening:
        magnitude = np.abs(product)
        product /= np.maximum(magnitude, np.finfo(magnitude.dtype).eps) ** whitening

    correlation = np.abs(fft.ifftn(product))
    yy, xx, allowed = _phase_grid(reference_fft.shape, tuple(max_shift))

    global_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = np.unravel_index(np.argmax(np.where(allowed, correlation, -np.inf)), correlation.shape)
    shift = np.array([yy[peak], xx[peak]], dtype=float)
    out_of_range = not allowed[global_peak]
    search_boundary = np.any(np.abs(shift) >= np.asarray(max_shift))

    away_from_peak = (np.abs(yy - shift[0]) > 2) | (np.abs(xx - shift[1]) > 2)
    second_peak = correlation[allowed & away_from_peak].max()
    peak_ratio = correlation[peak] / second_peak if second_peak else np.nan

    region_size = int(np.ceil(upsample * 1.5))
    centre = np.trunc(region_size / 2)
    offsets = centre - shift * upsample
    local = _local_dft(product.conj(), region_size, upsample, offsets).conj()
    local_peak = np.unravel_index(np.argmax(np.abs(local)), local.shape)
    shift += (np.asarray(local_peak) - centre) / upsample
    out_of_range |= np.any(np.abs(shift) > np.asarray(max_shift) + 1 / upsample)
    search_boundary |= np.any(np.abs(shift) >= np.asarray(max_shift) - 1 / upsample)
    return shift, float(peak_ratio), bool(out_of_range), bool(search_boundary)


def _prepare_reference(reference, max_shift_fraction=0.1, check_tiles=True):
    reference = np.asarray(reference)
    max_shift = np.maximum(
        1, np.round(np.asarray(reference.shape) * max_shift_fraction)).astype(int)
    parts = [(None, fft.fftn(_registration_image(reference)))]
    if check_tiles:
        tile_shape = np.maximum(32, np.round(np.asarray(reference.shape) * 0.6)).astype(int)
        for y in (0, reference.shape[0] - tile_shape[0]):
            for x in (0, reference.shape[1] - tile_shape[1]):
                tile = np.s_[y:y + tile_shape[0], x:x + tile_shape[1]]
                parts.append((tile, fft.fftn(_registration_image(reference[tile]))))
    # 15 August 2026: each pass reuses its fixed reference FFT; the second keeps four tiles
    return max_shift, parts


def _estimate_shift(reference, frame, *, upsample=10, whitening=0):
    max_shift, parts = reference
    frame = np.asarray(frame)
    estimates = []
    for tile, reference_fft in parts:
        image = frame if tile is None else frame[tile]
        estimates.append(_shift_from_fft(
            reference_fft,
            _registration_image(image),
            max_shift,
            upsample,
            whitening,
            ))

    shift, peak_ratio, out_of_range, search_boundary = estimates[0]
    tile_disagreement = np.nan
    if len(estimates) > 1:
        tile_shifts = np.asarray([estimate[0] for estimate in estimates[1:]])
        tile_disagreement = float(np.median(np.linalg.norm(tile_shifts - shift, axis=1)))
    return {
        'shift_y': float(shift[0]),
        'shift_x': float(shift[1]),
        'peak_ratio': peak_ratio,
        'tile_disagreement': tile_disagreement,
        'out_of_range': out_of_range,
        'search_boundary': search_boundary,
        }


def _estimate_shifts(reference, frames, *, upsample=10, whitening=0):
    estimate = partial(
        _estimate_shift, reference, upsample=upsample, whitening=whitening)
    # 15 August 2026: four frame jobs nearly halved reference time; FFT batches were slower
    # larger pools gave no stable gain on the ten-core benchmark machine
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(estimate, frames))


def estimate_shift(
        reference,
        frame,
        *,
        max_shift_fraction=0.1,
        upsample=10,
        whitening=0,
        check_tiles=True,
        ):
    prepared = _prepare_reference(reference, max_shift_fraction, check_tiles)
    return _estimate_shift(prepared, frame, upsample=upsample, whitening=whitening)


#%% local registration
def _tile_grid(shape, tile_size=64, stride=None):
    stride = tile_size // 2 if stride is None else stride
    starts_y = np.arange(0, shape[0] - tile_size + 1, stride, dtype=int)
    starts_x = np.arange(0, shape[1] - tile_size + 1, stride, dtype=int)
    if starts_y[-1] != shape[0] - tile_size:
        starts_y = np.r_[starts_y, shape[0] - tile_size]
    if starts_x[-1] != shape[1] - tile_size:
        starts_x = np.r_[starts_x, shape[1] - tile_size]
    tile_y, tile_x = np.meshgrid(starts_y, starts_x, indexing='ij')
    return {
        'start_y': tile_y.ravel(),
        'start_x': tile_x.ravel(),
        'tile_y': (tile_y + (tile_size - 1) / 2).ravel(),
        'tile_x': (tile_x + (tile_size - 1) / 2).ravel(),
        'n_y': len(starts_y),
        'n_x': len(starts_x),
        'tile_size': int(tile_size),
        'stride': int(stride),
        }


def _prepare_tile_reference(reference, tile_size=64, stride=None):
    reference = np.asarray(reference, dtype=np.float32)
    grid = _tile_grid(reference.shape, tile_size, stride)
    reference_image = _registration_image(reference)
    taper = _taper((tile_size, tile_size))
    reference_fft = []
    structure = []
    for y0, x0 in zip(grid['start_y'], grid['start_x']):
        tile = reference_image[y0:y0 + tile_size, x0:x0 + tile_size]
        reference_fft.append(fft.fftn(tile * taper))
        smooth = cv2.GaussianBlur(tile, (5, 5), 1, borderType=cv2.BORDER_REFLECT)
        gradient_y, gradient_x = np.gradient(smooth)
        structure.append([
            np.mean(gradient_y * gradient_y),
            np.mean(gradient_y * gradient_x),
            np.mean(gradient_x * gradient_x),
            ])
    return {
        **grid,
        'reference_fft': np.asarray(reference_fft, dtype=np.complex64),
        'structure': np.asarray(structure, dtype=np.float32),
        'shape': reference.shape,
        }


def _surface_entropy(surface):
    surface = np.asarray(surface, dtype=np.float64)
    scale = 1.4826 * np.median(np.abs(surface - np.median(surface)))
    scale = max(scale, np.finfo(np.float64).eps)
    probability = np.exp(np.clip((surface - surface.max()) / scale, -30, 0))
    probability /= probability.sum()
    return float(-np.sum(probability * np.log(probability)) / np.log(surface.size))


def _peak_curvature(surface, peak=None):
    peak_y, peak_x = (
        np.unravel_index(np.argmax(surface), surface.shape) if peak is None else peak)
    if peak_y in (0, surface.shape[0] - 1) or peak_x in (0, surface.shape[1] - 1):
        return np.full(3, np.nan, dtype=np.float32)
    centre = surface[peak_y, peak_x]
    curvature_yy = 2 * centre - surface[peak_y - 1, peak_x] - surface[peak_y + 1, peak_x]
    curvature_xx = 2 * centre - surface[peak_y, peak_x - 1] - surface[peak_y, peak_x + 1]
    curvature_yx = -0.25 * (
        surface[peak_y + 1, peak_x + 1]
        - surface[peak_y + 1, peak_x - 1]
        - surface[peak_y - 1, peak_x + 1]
        + surface[peak_y - 1, peak_x - 1]
        )
    return np.asarray([curvature_yy, curvature_yx, curvature_xx], dtype=np.float32)


def _tile_surface(
        reference_fft,
        tile,
        search_radius,
        upsample,
        whitening,
        selection_radius=None,
        ):
    product = reference_fft * fft.fftn(tile).conj()
    if whitening:
        magnitude = np.abs(product)
        product /= np.maximum(magnitude, np.finfo(magnitude.dtype).eps) ** whitening
    correlation = np.abs(fft.ifftn(product))
    shifts = np.arange(-search_radius, search_radius + 1)
    surface = correlation[np.ix_(shifts % tile.shape[0], shifts % tile.shape[1])]
    selected = surface
    if selection_radius is not None:
        selected = np.where(
            (np.abs(shifts[:, None]) <= selection_radius)
            & (np.abs(shifts[None, :]) <= selection_radius),
            surface,
            -np.inf,
            )
    peak_y, peak_x = np.unravel_index(np.argmax(selected), selected.shape)
    residual = np.asarray([
        shifts[peak_y],
        shifts[peak_x],
        ], dtype=float)
    region_size = int(np.ceil(upsample * 1.5))
    centre = np.trunc(region_size / 2)
    offsets = centre - residual * upsample
    local = _local_dft(
        product.conj(), region_size, upsample, offsets).conj()
    local_peak = np.unravel_index(np.argmax(np.abs(local)), local.shape)
    residual += (np.asarray(local_peak) - centre) / upsample
    away_from_peak = (
        (np.abs(np.arange(surface.shape[0])[:, None] - peak_y) > 1)
        | (np.abs(np.arange(surface.shape[1])[None, :] - peak_x) > 1)
        )
    second_peak = surface[away_from_peak].max()
    peak_ratio = surface[peak_y, peak_x] / second_peak if second_peak else np.nan
    boundary_radius = search_radius if selection_radius is None else selection_radius
    return {
        'surface': surface.astype(np.float32),
        'residual': residual,
        'peak_ratio': float(peak_ratio),
        'entropy': _surface_entropy(surface),
        'curvature': _peak_curvature(surface, (peak_y, peak_x)),
        'search_boundary': (
            abs(shifts[peak_y]) == boundary_radius
            or abs(shifts[peak_x]) == boundary_radius
            ),
        }


def _neighbour_residual(residual, n_y, n_x):
    field = np.asarray(residual, dtype=np.float32).reshape(n_y, n_x, 2)
    padded = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode='edge')
    result = np.empty((n_y, n_x), dtype=np.float32)
    for y in range(n_y):
        for x in range(n_x):
            neighbours = padded[y:y + 3, x:x + 3].reshape(-1, 2)
            neighbours = np.delete(neighbours, 4, axis=0)
            centre = np.median(neighbours, axis=0)
            spread = np.median(np.linalg.norm(neighbours - centre, axis=1)) + 0.1
            result[y, x] = np.linalg.norm(field[y, x] - centre) / spread
    return result.ravel()


def _shift_registration_image(frame, shift):
    frame = _registration_image(frame)
    matrix = np.array([[1, 0, shift[1]], [0, 1, shift[0]]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
        )


def _estimate_tile_shifts_from_image(
        prepared,
        frame_image,
        *,
        search_radius=3,
        upsample=10,
        whitening=0,
        prediction=None,
        selection_radius=None,
        ):
    if prediction is None:
        prediction = np.zeros((len(prepared['tile_y']), 2), dtype=np.float32)
    taper = _taper((prepared['tile_size'], prepared['tile_size']))
    surfaces = []
    residual = []
    peak_ratio = []
    entropy = []
    curvature = []
    search_boundary = []
    for tile_i, (y0, x0, reference_fft) in enumerate(zip(
            prepared['start_y'], prepared['start_x'], prepared['reference_fft'])):
        predicted_y, predicted_x = prediction[tile_i]
        # 17 August 2026: exact crops avoid another interpolation in the single-scale pass
        if predicted_y == 0 and predicted_x == 0:
            tile = frame_image[
                y0:y0 + prepared['tile_size'],
                x0:x0 + prepared['tile_size'],
                ]
        else:
            tile = cv2.getRectSubPix(
                frame_image,
                (prepared['tile_size'], prepared['tile_size']),
                (
                    float(prepared['tile_x'][tile_i] - predicted_x),
                    float(prepared['tile_y'][tile_i] - predicted_y),
                    ),
                )
        result = _tile_surface(
            reference_fft,
            tile * taper,
            search_radius,
            upsample,
            whitening,
            selection_radius,
            )
        surfaces.append(result['surface'])
        residual.append(result['residual'])
        peak_ratio.append(result['peak_ratio'])
        entropy.append(result['entropy'])
        curvature.append(result['curvature'])
        search_boundary.append(result['search_boundary'])
    incremental = np.asarray(residual, dtype=np.float32)
    residual = prediction + incremental
    curvature = np.asarray(curvature, dtype=np.float32)
    structure = prepared['structure']
    return {
        'tile_y': prepared['tile_y'].copy(),
        'tile_x': prepared['tile_x'].copy(),
        'residual_y': residual[:, 0],
        'residual_x': residual[:, 1],
        'predicted_residual_y': prediction[:, 0],
        'predicted_residual_x': prediction[:, 1],
        'incremental_residual_y': incremental[:, 0],
        'incremental_residual_x': incremental[:, 1],
        'peak_ratio': np.asarray(peak_ratio, dtype=np.float32),
        'surface': np.asarray(surfaces, dtype=np.float32),
        'surface_entropy': np.asarray(entropy, dtype=np.float32),
        'curvature_yy': curvature[:, 0],
        'curvature_yx': curvature[:, 1],
        'curvature_xx': curvature[:, 2],
        'structure_yy': structure[:, 0],
        'structure_yx': structure[:, 1],
        'structure_xx': structure[:, 2],
        'neighbour_residual': _neighbour_residual(
            residual, prepared['n_y'], prepared['n_x']),
        'search_boundary': np.asarray(search_boundary, dtype=bool),
        'search_radius': np.int16(search_radius),
        'selection_radius': np.int16(
            search_radius if selection_radius is None else selection_radius),
        'upsample': np.int16(upsample),
        'whitening': np.float32(whitening),
        }


def _estimate_tile_shifts(
        prepared,
        frame,
        rigid_shift,
        *,
        search_radius=3,
        upsample=10,
        whitening=0,
        ):
    result = _estimate_tile_shifts_from_image(
        prepared,
        _shift_registration_image(frame, rigid_shift),
        search_radius=search_radius,
        upsample=upsample,
        whitening=whitening,
        )
    result['rigid_shift_y'] = np.float32(rigid_shift[0])
    result['rigid_shift_x'] = np.float32(rigid_shift[1])
    return result


def _interpolate_tile_field(source, residual, target):
    source_y = source['tile_y'].reshape(source['n_y'], source['n_x'])[:, 0]
    source_x = source['tile_x'].reshape(source['n_y'], source['n_x'])[0]
    # 17 August 2026: extrapolation amplified noisy edge estimates; edge tiles use the nearest coarse value
    points = np.column_stack([
        np.clip(target['tile_y'], source_y[0], source_y[-1]),
        np.clip(target['tile_x'], source_x[0], source_x[-1]),
        ])
    return RegularGridInterpolator(
        (source_y, source_x),
        residual.reshape(source['n_y'], source['n_x']),
        )(points).astype(np.float32)


def _estimate_tile_shifts_coarse_to_fine(
        coarse,
        fine,
        frame,
        rigid_shift,
        *,
        search_radius=3,
        fine_radius=1,
        upsample=10,
        whitening=0,
        ):
    frame_image = _shift_registration_image(frame, rigid_shift)
    coarse_result = _estimate_tile_shifts_from_image(
        coarse,
        frame_image,
        search_radius=search_radius,
        upsample=upsample,
        whitening=whitening,
        )
    prediction = np.column_stack([
        _interpolate_tile_field(coarse, coarse_result['residual_y'], fine),
        _interpolate_tile_field(coarse, coarse_result['residual_x'], fine),
        ])
    # 17 August 2026: the surface stays 7 x 7; the coarse pass confines its fine peak to +/-1 px
    # each fine tile returns to the rigid image, which avoids another resampling step
    result = _estimate_tile_shifts_from_image(
        fine,
        frame_image,
        search_radius=search_radius,
        upsample=upsample,
        whitening=whitening,
        prediction=prediction,
        selection_radius=fine_radius,
        )
    result['rigid_shift_y'] = np.float32(rigid_shift[0])
    result['rigid_shift_x'] = np.float32(rigid_shift[1])
    result['coarse'] = coarse_result
    return result


def estimate_tile_shifts(
        reference,
        frame,
        rigid_shift,
        *,
        tile_size=64,
        stride=None,
        search_radius=3,
        upsample=10,
        whitening=0,
        ):
    prepared = _prepare_tile_reference(reference, tile_size, stride)
    return _estimate_tile_shifts(
        prepared,
        frame,
        rigid_shift,
        search_radius=search_radius,
        upsample=upsample,
        whitening=whitening,
        )


def estimate_tile_shifts_coarse_to_fine(
        reference,
        frame,
        rigid_shift,
        *,
        coarse_tile_size=128,
        tile_size=64,
        stride=None,
        search_radius=3,
        fine_radius=1,
        upsample=10,
        whitening=0,
        ):
    coarse = _prepare_tile_reference(
        reference, coarse_tile_size, coarse_tile_size // 2)
    fine = _prepare_tile_reference(reference, tile_size, stride)
    return _estimate_tile_shifts_coarse_to_fine(
        coarse,
        fine,
        frame,
        rigid_shift,
        search_radius=search_radius,
        fine_radius=fine_radius,
        upsample=upsample,
        whitening=whitening,
        )


#%% local field
def _surface_precision(surfaces):
    # 17 August 2026: full-surface covariance carries ambiguity and direction in one quantity
    shifts = np.arange(-(surfaces.shape[-1] // 2), surfaces.shape[-1] // 2 + 1)
    shift_y, shift_x = np.meshgrid(shifts, shifts, indexing='ij')
    coordinates = np.column_stack([shift_y.ravel(), shift_x.ravel()])
    precision = []
    for surface in surfaces:
        values = np.asarray(surface, dtype=np.float64).ravel()
        scale = 1.4826 * np.median(np.abs(values - np.median(values)))
        scale = max(scale, np.finfo(np.float64).eps)
        probability = np.exp(np.clip((values - values.max()) / scale, -30, 0))
        probability /= probability.sum()
        centre = probability @ coordinates
        difference = coordinates - centre
        covariance = (difference * probability[:, None]).T @ difference
        precision.append(np.linalg.inv(covariance + 0.01 * np.eye(2)))
    return np.asarray(precision, dtype=np.float32)


def _cubic_bspline(distance):
    distance = np.abs(np.asarray(distance, dtype=np.float64))
    basis = np.zeros_like(distance)
    central = distance < 1
    outer = (distance >= 1) & (distance < 2)
    basis[central] = (
        2 / 3 - distance[central] ** 2 + distance[central] ** 3 / 2)
    basis[outer] = (2 - distance[outer]) ** 3 / 6
    return basis


def _spline_grid(shape, spacing):
    control_y = np.arange(-spacing, shape[0] + 2 * spacing, spacing, dtype=float)
    control_x = np.arange(-spacing, shape[1] + 2 * spacing, spacing, dtype=float)
    return control_y, control_x


def _spline_basis(sample_y, sample_x, control_y, control_x, spacing):
    basis_y = _cubic_bspline(
        (np.asarray(sample_y)[:, None] - control_y[None]) / spacing)
    basis_x = _cubic_bspline(
        (np.asarray(sample_x)[:, None] - control_x[None]) / spacing)
    return np.einsum('iy,ix->iyx', basis_y, basis_x).reshape(len(basis_y), -1)


def _spline_penalty(n_y, n_x, magnitude=0.01):
    identity_y = np.eye(n_y)
    identity_x = np.eye(n_x)
    first_y = np.diff(identity_y, axis=0)
    first_x = np.diff(identity_x, axis=0)
    second_y = np.diff(identity_y, n=2, axis=0)
    second_x = np.diff(identity_x, n=2, axis=0)
    dyy = np.kron(second_y, identity_x)
    dxx = np.kron(identity_y, second_x)
    dyx = np.kron(first_y, first_x)
    return (
        dyy.T @ dyy
        + 2 * dyx.T @ dyx
        + dxx.T @ dxx
        + magnitude * np.eye(n_y * n_x)
        )


def _tile_field_evidence(tiles, accepted=None):
    precision = _surface_precision(tiles['surface'])
    if accepted is None:
        # 17 August 2026: 2 is the normalised-median PIV limit; it stays fixed for held-out work
        accepted = (
            (tiles['neighbour_residual'] <= 2)
            & ~tiles['search_boundary']
            & (tiles['peak_ratio'] >= 1)
            )
    scale = (
        np.median(np.trace(precision[accepted], axis1=1, axis2=2))
        if accepted.any() else 1)
    return {
        'accepted': np.asarray(accepted, dtype=bool),
        'precision': precision / scale,
        }


def fit_tile_field(
        tiles,
        shape,
        tile_size,
        *,
        spatial_penalty=1,
        magnitude_penalty=0.01,
        accepted=None,
        evidence=None,
        ):
    residual = np.column_stack([tiles['residual_y'], tiles['residual_x']])
    evidence = _tile_field_evidence(tiles, accepted) if evidence is None else evidence
    accepted = evidence['accepted']
    precision = evidence['precision']

    control_y, control_x = _spline_grid(shape, tile_size)
    basis = _spline_basis(
        tiles['tile_y'], tiles['tile_x'], control_y, control_x, tile_size)
    n_coefficients = basis.shape[1]
    n_parameters = 2 + 2 * n_coefficients
    parameters = np.zeros(n_parameters, dtype=np.float64)
    if np.mean(accepted) >= 0.60:
        normal = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        target = np.zeros(n_parameters, dtype=np.float64)
        for tile_i in np.flatnonzero(accepted):
            design = np.zeros((2, n_parameters), dtype=np.float64)
            design[0, 0] = 1
            design[1, 1] = 1
            design[0, 2:2 + n_coefficients] = basis[tile_i]
            design[1, 2 + n_coefficients:] = basis[tile_i]
            weight = precision[tile_i]
            normal += design.T @ weight @ design
            target += design.T @ weight @ residual[tile_i]

        # 17 August 2026: bending scales with inverse control-point area
        # magnitude shrinkage leaves shared movement with the unpenalised global adjustment
        penalty = (
            spatial_penalty
            * _spline_penalty(
                len(control_y), len(control_x), magnitude=magnitude_penalty)
            / tile_size ** 2
            )
        normal[2:2 + n_coefficients, 2:2 + n_coefficients] += penalty
        normal[2 + n_coefficients:, 2 + n_coefficients:] += penalty
        parameters = np.linalg.solve(normal, target)
    global_shift = parameters[:2]
    coefficient_y = parameters[2:2 + n_coefficients]
    coefficient_x = parameters[2 + n_coefficients:]
    predicted = np.column_stack([
        global_shift[0] + basis @ coefficient_y,
        global_shift[1] + basis @ coefficient_x,
        ])
    return {
        'global_shift_y': np.float32(global_shift[0]),
        'global_shift_x': np.float32(global_shift[1]),
        'coefficient_y': coefficient_y.reshape(len(control_y), len(control_x)).astype(np.float32),
        'coefficient_x': coefficient_x.reshape(len(control_y), len(control_x)).astype(np.float32),
        'control_y': control_y.astype(np.float32),
        'control_x': control_x.astype(np.float32),
        'tile_y': tiles['tile_y'].copy(),
        'tile_x': tiles['tile_x'].copy(),
        'predicted_y': predicted[:, 0].astype(np.float32),
        'predicted_x': predicted[:, 1].astype(np.float32),
        'accepted': np.asarray(accepted, dtype=bool),
        'precision_yy': precision[:, 0, 0],
        'precision_yx': precision[:, 0, 1],
        'precision_xx': precision[:, 1, 1],
        'spatial_penalty': np.float32(spatial_penalty),
        'magnitude_penalty': np.float32(magnitude_penalty),
        'tile_size': np.int16(tile_size),
        }


def refine_tile_field(
        tiles,
        shape,
        tile_size,
        *,
        spatial_penalty=10,
        magnitude_penalty=1,
        residual_limit=0.28,
        evidence=None,
        initial=None,
        ):
    evidence = _tile_field_evidence(tiles) if evidence is None else evidence
    if initial is None:
        initial = fit_tile_field(
            tiles,
            shape,
            tile_size,
            spatial_penalty=spatial_penalty,
            magnitude_penalty=magnitude_penalty,
            evidence=evidence,
            )
    difference = np.column_stack([
        initial['predicted_y'] - tiles['residual_y'],
        initial['predicted_x'] - tiles['residual_x'],
        ])
    field_residual = np.sqrt(np.einsum(
        'ni,nij,nj->n', difference, evidence['precision'], difference))
    refined_evidence = {
        'accepted': evidence['accepted'] & (field_residual <= residual_limit),
        'precision': evidence['precision'],
        }
    refined = fit_tile_field(
        tiles,
        shape,
        tile_size,
        spatial_penalty=spatial_penalty,
        magnitude_penalty=magnitude_penalty,
        evidence=refined_evidence,
        )
    refined['field_residual'] = field_residual.astype(np.float32)
    refined['residual_limit'] = np.float32(residual_limit)
    return refined


def evaluate_tile_field(field, sample_y, sample_x):
    basis = _spline_basis(
        np.asarray(sample_y).ravel(),
        np.asarray(sample_x).ravel(),
        field['control_y'],
        field['control_x'],
        int(field['tile_size']),
        )
    shift_y = field['global_shift_y'] + basis @ field['coefficient_y'].ravel()
    shift_x = field['global_shift_x'] + basis @ field['coefficient_x'].ravel()
    return shift_y.reshape(np.shape(sample_y)), shift_x.reshape(np.shape(sample_x))


def tile_field_image(field, shape):
    sample_y = np.arange(shape[0])
    sample_x = np.arange(shape[1])
    basis_y = _cubic_bspline(
        (sample_y[:, None] - field['control_y'][None]) / int(field['tile_size']))
    basis_x = _cubic_bspline(
        (sample_x[:, None] - field['control_x'][None]) / int(field['tile_size']))
    shift_y = (
        field['global_shift_y']
        + basis_y @ field['coefficient_y'] @ basis_x.T)
    shift_x = (
        field['global_shift_x']
        + basis_y @ field['coefficient_x'] @ basis_x.T)
    return shift_y.astype(np.float32), shift_x.astype(np.float32)


def assess_tile_field(field, shape, *, focal_loss=False):
    shift_y, shift_x = tile_field_image(field, shape)
    dy_dy, dy_dx = np.gradient(shift_y)
    dx_dy, dx_dx = np.gradient(shift_x)
    # 17 August 2026: sampling uses p-u(p); its Jacobian carries the same minus sign
    jacobian = (1 - dy_dy) * (1 - dx_dx) - dy_dx * dx_dy
    tile_y = field['tile_y']
    tile_x = field['tile_x']
    n_y = len(np.unique(tile_y))
    n_x = len(np.unique(tile_x))
    neighbour_difference = np.nan
    if n_y >= 3 and n_x >= 3:
        predicted = np.stack([
            field['predicted_y'].reshape(n_y, n_x),
            field['predicted_x'].reshape(n_y, n_x),
            ], axis=-1)
        neighbour_difference = max(
            np.linalg.norm(np.diff(predicted, axis=0), axis=-1).max(),
            np.linalg.norm(np.diff(predicted, axis=1), axis=-1).max(),
            )
    maximum = float(np.hypot(shift_y, shift_x).max())
    accepted_fraction = float(np.mean(field['accepted']))
    reason = 'accepted'
    if n_y < 3 or n_x < 3 or accepted_fraction < 0.60:
        reason = 'insufficient_tiles'
    elif focal_loss:
        reason = 'focal_loss'
    elif maximum > 3:
        reason = 'field_overshoot'
    elif neighbour_difference > 3:
        reason = 'neighbour_disagreement'
    elif jacobian.min() < 0.80 or jacobian.max() > 1.25:
        reason = 'jacobian_limit'
    return {
        'model_used': 'piecewise_rigid' if reason == 'accepted' else 'rigid',
        'fallback_reason': reason,
        'accepted_tile_fraction': accepted_fraction,
        'field_rms_px': float(np.sqrt(np.mean(shift_y ** 2 + shift_x ** 2))),
        'field_max_px': maximum,
        'neighbour_difference_max_px': float(neighbour_difference),
        'jacobian_min': float(jacobian.min()),
        'jacobian_max': float(jacobian.max()),
        }


def field_coordinates(field, shape, rigid_shift=(0, 0), static_offset=(0, 0)):
    residual_y, residual_x = tile_field_image(field, shape)
    y, x = np.indices(shape, dtype=np.float32)
    correction_y = rigid_shift[0] + residual_y + static_offset[0]
    correction_x = rigid_shift[1] + residual_x + static_offset[1]
    source_y = y - correction_y
    source_x = x - correction_x
    valid = (
        (source_y >= 0) & (source_y <= shape[0] - 1)
        & (source_x >= 0) & (source_x <= shape[1] - 1)
        )
    rows, columns = np.nonzero(valid)
    bounds = (
        int(rows.min()), int(rows.max() + 1),
        int(columns.min()), int(columns.max() + 1),
        )
    return source_y, source_x, valid, bounds


def warp_frame_piecewise(frame, field, rigid_shift=(0, 0), static_offset=(0, 0)):
    source_y, source_x, valid, _ = field_coordinates(
        field, frame.shape, rigid_shift, static_offset)
    registered = cv2.remap(
        np.asarray(frame, dtype=np.float32),
        source_x,
        source_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
        )
    # 19 August 2026: OpenCV's 1/32-pixel interpolation can sample the border just inside an edge
    valid &= np.isfinite(registered)
    registered[~valid] = np.nan
    rows, columns = np.nonzero(valid)
    bounds = (
        int(rows.min()), int(rows.max() + 1),
        int(columns.min()), int(columns.max() + 1),
        )
    return registered, bounds, valid


def select_registration_model(rigid, piecewise):
    thresholds = {
        'gradient_ncc_gain': 0.01,
        'residual_p95_gain_px': 0.15,
        'valid_fraction_loss': 0.02,
        'cross_channel_worsening_px': 0.05,
        'accepted_or_fallback_fraction': 0.95,
        }
    comparison = {
        'gradient_ncc_gain': (
            piecewise['gradient_ncc'] - rigid['gradient_ncc']),
        'residual_p95_gain_px': (
            rigid['residual_p95_px'] - piecewise['residual_p95_px']),
        'valid_fraction_loss': (
            rigid['valid_fraction'] - piecewise['valid_fraction']),
        'cross_channel_worsening_px': (
            piecewise['cross_channel_residual_px']
            - rigid['cross_channel_residual_px']),
        'accepted_or_fallback_fraction': piecewise['accepted_or_fallback_fraction'],
        }
    passed = {
        'gradient_ncc': (
            comparison['gradient_ncc_gain'] >= thresholds['gradient_ncc_gain']),
        'residual_p95': (
            comparison['residual_p95_gain_px'] >= thresholds['residual_p95_gain_px']),
        'valid_fraction': (
            comparison['valid_fraction_loss'] <= thresholds['valid_fraction_loss']),
        'cross_channel': (
            comparison['cross_channel_worsening_px']
            <= thresholds['cross_channel_worsening_px']),
        'fallback_coverage': (
            comparison['accepted_or_fallback_fraction']
            >= thresholds['accepted_or_fallback_fraction']),
        }
    return {
        'selected_model': 'piecewise_rigid' if all(passed.values()) else 'rigid',
        'comparison': comparison,
        'thresholds': thresholds,
        'passed': passed,
        }


def _valid_bounds(shape, shift_y, shift_x):
    y0 = max(0, int(np.ceil(shift_y)))
    y1 = min(shape[0], int(np.floor(shape[0] - 1 + shift_y)) + 1)
    x0 = max(0, int(np.ceil(shift_x)))
    x1 = min(shape[1], int(np.floor(shape[1] - 1 + shift_x)) + 1)
    return y0, y1, x0, x1


def warp_frame(frame, shift_y, shift_x):
    registered = ndi.shift(
        np.asarray(frame, dtype=np.float32),
        (shift_y, shift_x),
        order=1,
        mode='constant',
        cval=np.nan,
        prefilter=False,
        )
    return registered, _valid_bounds(frame.shape, shift_y, shift_x)


def _registered_mean(frames, shifts, accepted, shape):
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint16)
    for frame, shift, use_frame in zip(frames, shifts, accepted):
        if not use_frame:
            continue
        registered, _ = warp_frame(frame, *shift)
        valid = np.isfinite(registered)
        total[valid] += registered[valid]
        count[valid] += 1
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)


def _reference_bounds(shape, shift_y, shift_x):
    # Lanczos4 samples floor(x)-3 through floor(x)+4
    y0 = max(0, int(np.ceil(shift_y + 3)))
    y1 = min(shape[0], int(np.ceil(shape[0] - 4 + shift_y)))
    x0 = max(0, int(np.ceil(shift_x + 3)))
    x1 = min(shape[1], int(np.ceil(shape[1] - 4 + shift_x)))
    return y0, y1, x0, x1


def _reference_frame(frame, shift_y, shift_x):
    # 15 August 2026: the interpolation ablation favoured Lanczos reference copies;
    # registered movies stay bilinear
    matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    image = cv2.warpAffine(
        np.asarray(frame, dtype=np.float32),
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
        )
    return image, _reference_bounds(frame.shape, shift_y, shift_x)


def _reference_mean(frames, shifts, accepted, shape):
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint16)
    for frame, shift, use_frame in zip(frames, shifts, accepted):
        if not use_frame:
            continue
        image, bounds = _reference_frame(frame, *shift)
        y0, y1, x0, x1 = bounds
        total[y0:y1, x0:x1] += image[y0:y1, x0:x1]
        count[y0:y1, x0:x1] += 1
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)


def make_reference(
        frames,
        *,
        max_frames=1000,
        min_frames=500,
        whitening=0,
        min_peak_ratio=1.1,
        max_tile_disagreement=1.0,
        ):
    n_frames = len(frames)
    if n_frames < 50:
        raise ValueError('at least 50 frames are required for the reference')

    target_frames = min(min_frames, n_frames)
    fallback = target_frames < min_frames

    sample_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    first_frame = np.asarray(frames[int(sample_indices[0])])
    shape = first_frame.shape
    dtype = first_frame.dtype
    # 16 August 2026: indexing the source avoids carrying a 0.49 GiB TIFF copy through both passes
    sample = (np.asarray(frames[int(frame_i)]) for frame_i in sample_indices)
    median = np.empty(len(sample_indices), dtype=np.float32)
    mad = np.empty(len(sample_indices), dtype=np.float32)
    gradient = np.empty(len(sample_indices), dtype=np.float32)
    saturation = np.zeros(len(sample_indices), dtype=np.float32)
    limits = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    for frame_i, frame in enumerate(sample):
        image = np.asarray(frame, dtype=np.float32)
        median[frame_i] = np.median(image)
        mad[frame_i] = np.median(np.abs(image - median[frame_i]))
        gradient[frame_i] = np.mean(
            np.hypot(ndi.sobel(image, axis=0), ndi.sobel(image, axis=1)))
        if limits is not None:
            saturation[frame_i] = np.mean((frame == limits.min) | (frame == limits.max))

    informative = (mad > 0) & (saturation <= 0.01)
    if informative.sum() < target_frames:
        if informative.sum() < 50:
            raise ValueError('fewer than 50 informative frames remain')
        target_frames = int(informative.sum())
        fallback = True
    informative &= gradient >= 0.1 * np.median(gradient[informative])
    if informative.sum() < target_frames:
        if informative.sum() < 50:
            raise ValueError('fewer than 50 informative frames remain')
        target_frames = int(informative.sum())
        fallback = True

    candidates = np.flatnonzero(informative)
    anchor_candidates = candidates[
        np.linspace(0, len(candidates) - 1, min(100, len(candidates)), dtype=int)
        ]
    small = np.stack([
        _registration_image(frames[int(sample_indices[i])])[::4, ::4].ravel()
        for i in anchor_candidates
        ])
    small -= small.mean(axis=1, keepdims=True)
    small /= np.linalg.norm(small, axis=1, keepdims=True)
    similarity = small @ small.T
    anchor = anchor_candidates[np.argmax(np.median(similarity, axis=1))]

    first_shifts = np.full((len(sample_indices), 2), np.nan, dtype=np.float32)
    first_peak_ratio = np.full(len(sample_indices), np.nan, dtype=np.float32)
    first_out_of_range = np.zeros(len(sample_indices), dtype=bool)
    first_accepted = np.zeros(len(sample_indices), dtype=bool)
    first_reference = _prepare_reference(
        frames[int(sample_indices[anchor])], check_tiles=False)
    sampled_frames = (
        frames[int(sample_indices[frame_i])] for frame_i in candidates)
    estimates = _estimate_shifts(first_reference, sampled_frames, whitening=whitening)
    for frame_i, estimate in zip(candidates, estimates):
        first_shifts[frame_i] = estimate['shift_y'], estimate['shift_x']
        first_peak_ratio[frame_i] = estimate['peak_ratio']
        first_out_of_range[frame_i] = estimate['out_of_range']
        first_accepted[frame_i] = (
            estimate['peak_ratio'] >= min_peak_ratio and not estimate['out_of_range']
            )
    if first_accepted.sum() < target_frames:
        # 18 August 2026: weak-texture references get one relaxed peak-ratio pass
        # the second pass still records confidence
        relaxed_peak_ratio = max(1.02, min_peak_ratio - 0.07)
        first_accepted = (
            first_peak_ratio >= relaxed_peak_ratio
            ) & ~first_out_of_range
        if first_accepted.sum() < 50:
            raise ValueError('fewer than 50 frames align with the first reference')
        target_frames = min(target_frames, int(first_accepted.sum()))
        fallback = True
    sampled_frames = (frames[int(frame_i)] for frame_i in sample_indices)
    provisional = _registered_mean(
        sampled_frames, first_shifts, first_accepted, shape)

    shifts = np.full((len(sample_indices), 2), np.nan, dtype=np.float32)
    peak_ratio = np.full(len(sample_indices), np.nan, dtype=np.float32)
    tile_disagreement = np.full(len(sample_indices), np.nan, dtype=np.float32)
    second_out_of_range = np.zeros(len(sample_indices), dtype=bool)
    aligned = np.zeros(len(sample_indices), dtype=bool)
    second_reference = _prepare_reference(provisional)
    sampled_frames = (
        frames[int(sample_indices[frame_i])] for frame_i in candidates)
    estimates = _estimate_shifts(second_reference, sampled_frames, whitening=whitening)
    for frame_i, estimate in zip(candidates, estimates):
        shifts[frame_i] = estimate['shift_y'], estimate['shift_x']
        peak_ratio[frame_i] = estimate['peak_ratio']
        tile_disagreement[frame_i] = estimate['tile_disagreement']
        second_out_of_range[frame_i] = estimate['out_of_range']
        aligned[frame_i] = (
            estimate['peak_ratio'] >= min_peak_ratio
            and estimate['tile_disagreement'] <= max_tile_disagreement
            and not estimate['out_of_range']
            )
    if aligned.sum() < target_frames:
        # 18 August 2026: the provisional image can inherit weak texture from the first pass
        aligned = (
            (peak_ratio >= max(1.02, min_peak_ratio - 0.07))
            & (tile_disagreement <= max(2.0, max_tile_disagreement * 2))
            & ~second_out_of_range
            )
        if aligned.sum() < 50:
            raise ValueError('fewer than 50 frames align with the provisional reference')
        target_frames = min(target_frames, int(aligned.sum()))
        fallback = True

    bounds = np.asarray([
        _reference_bounds(shape, *shifts[frame_i])
        for frame_i in np.flatnonzero(aligned)
        ])
    common = np.s_[
        bounds[:, 0].max():bounds[:, 1].min(),
        bounds[:, 2].max():bounds[:, 3].min(),
        ]
    aligned_frames = np.flatnonzero(aligned)
    crop_shape = (
        common[0].stop - common[0].start,
        common[1].stop - common[1].start,
        )
    copies = np.empty((len(aligned_frames), *crop_shape), dtype=np.float32)
    # 16 August 2026: fill one array directly; crop views kept their full warped frames alive
    for copy_i, frame_i in enumerate(aligned_frames):
        image, _ = _reference_frame(
            frames[int(sample_indices[frame_i])], *shifts[frame_i])
        copies[copy_i] = image[common]
    consensus = np.empty(crop_shape, dtype=np.float32)
    # 16 August 2026: narrow strips keep NumPy's median copy below 32 MiB
    for x0 in range(0, crop_shape[1], 32):
        x1 = min(x0 + 32, crop_shape[1])
        consensus[:, x0:x1] = np.median(copies[:, :, x0:x1], axis=0)
    consensus_gradient = np.hypot(
        ndi.sobel(consensus, axis=0), ndi.sobel(consensus, axis=1)).ravel()
    consensus_gradient -= consensus_gradient.mean()
    consensus_power = np.dot(consensus_gradient, consensus_gradient)
    correlation = np.full(len(sample_indices), np.nan, dtype=np.float32)
    for copy, frame_i in zip(copies, aligned_frames):
        copy_gradient = np.hypot(
            ndi.sobel(copy, axis=0), ndi.sobel(copy, axis=1)).ravel()
        copy_gradient -= copy_gradient.mean()
        correlation[frame_i] = np.dot(consensus_gradient, copy_gradient) / np.sqrt(
            consensus_power * np.dot(copy_gradient, copy_gradient))

    # 15 August 2026: 20 bins retain the recording span; synthetic focal changes ranked below the consensus
    accepted = np.zeros(len(sample_indices), dtype=bool)
    time_bins = np.array_split(np.arange(len(sample_indices)), min(20, target_frames))
    n_per_bin = np.full(len(time_bins), target_frames // len(time_bins))
    n_per_bin[:target_frames % len(time_bins)] += 1
    for frames_in_bin, n_keep in zip(time_bins, n_per_bin):
        frames_in_bin = frames_in_bin[aligned[frames_in_bin]]
        order = frames_in_bin[np.argsort(correlation[frames_in_bin])]
        accepted[order[-n_keep:]] = True

    n_missing = target_frames - accepted.sum()
    if n_missing:
        remaining = np.flatnonzero(aligned & ~accepted)
        order = remaining[np.argsort(correlation[remaining])]
        accepted[order[-n_missing:]] = True

    sampled_frames = (frames[int(frame_i)] for frame_i in sample_indices)
    reference = _reference_mean(sampled_frames, shifts, accepted, shape)
    return {
        'image': reference,
        'sample_indices': sample_indices,
        'aligned': aligned,
        'accepted': accepted,
        'shift_y': shifts[:, 0],
        'shift_x': shifts[:, 1],
        'peak_ratio': peak_ratio,
        'tile_disagreement': tile_disagreement,
        'reference_correlation': correlation,
        'gradient_information': gradient,
        'saturation_fraction': saturation,
        'reference_fallback': fallback,
        'reference_target_frames': np.int64(target_frames),
        'reference_aligned_count': np.int64(aligned.sum()),
        'reference_accepted_count': np.int64(accepted.sum()),
        }


def _gradient_image(image):
    return np.hypot(ndi.sobel(image, axis=0), ndi.sobel(image, axis=1))


def _ncc(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    power = np.sqrt(np.dot(first, first) * np.dot(second, second))
    return float(np.dot(first, second) / power) if power else np.nan


def _gradient_ncc(first, second):
    return _ncc(_gradient_image(first), _gradient_image(second))


#%% motion and focal QC
def _high_frequency_image(image):
    valid = np.isfinite(image)
    if not valid.any():
        return np.full(image.shape, np.nan, dtype=np.float32)
    filled = np.where(valid, image, np.nanmedian(image))
    # 17 August 2026: the second filter gives an effective 2.5 px scale after
    # the 1.5 px camera-noise smoothing
    denoised = ndi.gaussian_filter(filled, 1.5)
    return denoised - ndi.gaussian_filter(denoised, 2)


def _high_frequency_fraction(reference_detail, image):
    valid = np.isfinite(image)
    interior = cv2.erode(
        valid.astype(np.uint8),
        np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        iterations=8,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
        ).astype(bool)
    if not interior.any():
        return np.nan
    image_detail = _high_frequency_image(image)
    reference_power = np.var(reference_detail[interior])
    return float(np.var(image_detail[interior]) / reference_power) if reference_power else np.nan


def _spatial_correlation(image):
    pairs = (
        (image[:, :-1], image[:, 1:]),
        (image[:-1], image[1:]),
        )
    correlations = []
    for first, second in pairs:
        valid = np.isfinite(first) & np.isfinite(second)
        if not valid.any():
            correlations.append(np.nan)
            continue
        first = first[valid] - first[valid].mean()
        second = second[valid] - second[valid].mean()
        power = np.sqrt(np.dot(first, first) * np.dot(second, second))
        correlations.append(np.dot(first, second) / power if power else np.nan)
    correlations = np.asarray(correlations)
    return float(np.nanmean(correlations)) if np.isfinite(correlations).any() else np.nan


def _gain_offset(reference, image):
    valid = np.isfinite(reference) & np.isfinite(image)
    if not valid.any():
        return np.nan, np.nan
    x = reference[valid]
    y = image[valid]
    x_mean = x.mean()
    y_mean = y.mean()
    variance = np.dot(x - x_mean, x - x_mean)
    gain = np.dot(x - x_mean, y - y_mean) / variance if variance else np.nan
    return float(gain), float(y_mean - gain * x_mean)


def _low_threshold(values, baseline, minimum_drop, n_mad):
    values = values[baseline & np.isfinite(values)]
    if not len(values):
        return np.nan
    centre = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - centre))
    return centre - max(minimum_drop, n_mad * mad)


def _high_threshold(values, baseline, minimum_rise, n_mad):
    values = values[baseline & np.isfinite(values)]
    if not len(values):
        return np.nan
    centre = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - centre))
    return centre + max(minimum_rise, n_mad * mad)


def _quality_thresholds(fields, baseline, focal_mads):
    thresholds = {
        'canonical_focal': _low_threshold(
            fields['canonical_gradient_ncc'], baseline, 0.15, focal_mads),
        'local_focal': _low_threshold(
            fields['local_gradient_ncc'], baseline, 0.15, focal_mads),
        'high_frequency_focal': _low_threshold(
            fields['high_frequency_fraction'], baseline, 0.08, focal_mads),
        'canonical_ambiguous': _low_threshold(
            fields['canonical_gradient_ncc'], baseline, 0.08, 3),
        }
    gain = fields['control_gain']
    gain_values = gain[baseline & np.isfinite(gain)]
    gain_threshold = _low_threshold(gain, baseline, 0, focal_mads)
    gain_centre = np.median(gain_values) if len(gain_values) else np.nan
    thresholds['control_gain_focal'] = (
        max(0.2 * gain_centre, gain_threshold)
        if np.isfinite(gain_threshold) and np.isfinite(gain_centre)
        else np.nan
        )
    return thresholds


def _focal_candidates(fields, thresholds):
    return (
        (fields['canonical_gradient_ncc'] < thresholds['canonical_focal'])
        & (fields['local_gradient_ncc'] < thresholds['local_focal'])
        & (fields['high_frequency_fraction'] < thresholds['high_frequency_focal'])
        & (fields['control_gain'] < thresholds['control_gain_focal'])
        )


def make_local_references(
        reference,
        frames,
        estimates,
        sampling_frequency_hz,
        *,
        timestamps=None,
        window_s=60,
        max_frames=100,
        ):
    timestamps = (
        np.arange(len(frames), dtype=float) / sampling_frequency_hz
        if timestamps is None else np.asarray(timestamps, dtype=float)
        )
    window_index = np.floor((timestamps - timestamps[0]) / window_s).astype(np.int64)
    reference_index = np.empty(len(frames), dtype=np.int32)
    references = []
    canonical_fallback = []
    for window_i in np.unique(window_index):
        frame_indices = np.flatnonzero(window_index == window_i)
        candidates = np.asarray([
            frame_i for frame_i in frame_indices
            if estimates[frame_i]['peak_ratio'] >= 1.1
            and estimates[frame_i]['tile_disagreement'] <= 1.0
            and not estimates[frame_i]['out_of_range']
            ])
        if not len(candidates):
            # no credible estimate in this interval; use the canonical reference
            references.append(np.asarray(reference, dtype=np.float32))
            canonical_fallback.append(True)
            reference_index[frame_indices] = len(references) - 1
            continue
        selected = candidates[
            np.linspace(0, len(candidates) - 1, min(max_frames, len(candidates)), dtype=int)
            ]
        bounds = np.asarray([
            _valid_bounds(
                frames[frame_i].shape,
                estimates[frame_i]['shift_y'], estimates[frame_i]['shift_x'])
            for frame_i in selected
            ])
        common = np.s_[
            bounds[:, 0].max():bounds[:, 1].min(),
            bounds[:, 2].max():bounds[:, 3].min(),
            ]
        copies = np.empty(
            (len(selected),
             common[0].stop - common[0].start,
             common[1].stop - common[1].start),
            dtype=np.float32,
            )
        for copy_i, frame_i in enumerate(selected):
            registered, _ = warp_frame(
                frames[frame_i],
                estimates[frame_i]['shift_y'], estimates[frame_i]['shift_x'])
            copies[copy_i] = registered[common]
        local_reference = np.full(frames[0].shape, np.nan, dtype=np.float32)
        local_reference[common] = np.median(copies, axis=0)
        references.append(local_reference)
        canonical_fallback.append(False)
        reference_index[frame_indices] = len(references) - 1
    return {
        'images': np.asarray(references),
        'gradient_images': np.asarray([
            _gradient_image(image) for image in references]),
        'reference_index': reference_index,
        'canonical_fallback': np.asarray(canonical_fallback, dtype=bool),
        'window': np.unique(window_index),
        }


def _temporal_difference(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    first = first[valid]
    second = second[valid]
    first_centre = np.median(first)
    second_centre = np.median(second)
    first_mad = np.median(np.abs(first - first_centre))
    second_mad = np.median(np.abs(second - second_centre))
    if not first_mad or not second_mad:
        return np.nan
    first = (first - first_centre) / first_mad
    second = (second - second_centre) / second_mad
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _new_quality_fields(n_frames, timestamps, sampling_frequency_hz):
    frame_period = 1 / sampling_frequency_hz
    timing_fault = np.zeros(n_frames, dtype=bool)
    timing_fault[1:] = np.abs(np.diff(timestamps) - frame_period) > frame_period / 2
    return {
        'time_s': timestamps,
        'dx_px': np.empty(n_frames, dtype=np.float32),
        'dy_px': np.empty(n_frames, dtype=np.float32),
        'displacement_magnitude_px': np.empty(n_frames, dtype=np.float32),
        'peak_ratio': np.empty(n_frames, dtype=np.float32),
        'tile_disagreement_px': np.empty(n_frames, dtype=np.float32),
        'canonical_gradient_ncc': np.empty(n_frames, dtype=np.float32),
        'local_gradient_ncc': np.empty(n_frames, dtype=np.float32),
        'high_frequency_fraction': np.empty(n_frames, dtype=np.float32),
        'spatial_correlation': np.empty(n_frames, dtype=np.float32),
        'temporal_difference': np.full(n_frames, np.nan, dtype=np.float32),
        'control_gain': np.empty(n_frames, dtype=np.float32),
        'control_offset': np.empty(n_frames, dtype=np.float32),
        'signal_gain': np.full(n_frames, np.nan, dtype=np.float32),
        'signal_offset': np.full(n_frames, np.nan, dtype=np.float32),
        'valid_pixel_fraction': np.empty(n_frames, dtype=np.float32),
        'search_boundary': np.empty(n_frames, dtype=bool),
        'detector_artifact': np.empty(n_frames, dtype=bool),
        'timing_fault': timing_fault,
        'photometric_control_gain_change': np.full(n_frames, np.nan, dtype=np.float32),
        'photometric_control_offset_change': np.full(n_frames, np.nan, dtype=np.float32),
        'photometric_signal_gain_change': np.full(n_frames, np.nan, dtype=np.float32),
        'photometric_signal_offset_change': np.full(n_frames, np.nan, dtype=np.float32),
        'photometric_artifact': np.empty(n_frames, dtype=bool),
        'local_reference_fallback': np.empty(n_frames, dtype=bool),
        'recommended_state': np.empty(n_frames, dtype=object),
        'reason_code': np.empty(n_frames, dtype=object),
        '_outside_search': np.empty(n_frames, dtype=bool),
        }


def _measure_registered_quality(
        fields,
        frame_i,
        raw_control,
        registered_control,
        estimate,
        reference,
        reference_detail,
        reference_gradient,
        local_references,
        previous_registered,
        registered_signal=None,
        signal_reference=None,
        ):
    local_i = local_references['reference_index'][frame_i]
    gain, offset = _gain_offset(reference, registered_control)
    valid = np.isfinite(registered_control)
    limits = (
        np.iinfo(raw_control.dtype)
        if np.issubdtype(raw_control.dtype, np.integer) else None)
    saturation = (
        np.mean((raw_control == limits.min) | (raw_control == limits.max))
        if limits is not None else 0)

    fields['dx_px'][frame_i] = estimate['shift_x']
    fields['dy_px'][frame_i] = estimate['shift_y']
    fields['displacement_magnitude_px'][frame_i] = np.hypot(
        estimate['shift_y'], estimate['shift_x'])
    fields['peak_ratio'][frame_i] = estimate['peak_ratio']
    fields['tile_disagreement_px'][frame_i] = estimate['tile_disagreement']
    registered_gradient = _gradient_image(registered_control)
    fields['canonical_gradient_ncc'][frame_i] = _ncc(
        reference_gradient, registered_gradient)
    fields['local_gradient_ncc'][frame_i] = _ncc(
        local_references['gradient_images'][local_i], registered_gradient)
    fields['high_frequency_fraction'][frame_i] = _high_frequency_fraction(
        reference_detail, registered_control)
    fields['spatial_correlation'][frame_i] = _spatial_correlation(registered_control)
    if previous_registered is not None:
        fields['temporal_difference'][frame_i] = _temporal_difference(
            previous_registered, registered_control)
    fields['control_gain'][frame_i] = gain
    fields['control_offset'][frame_i] = offset
    if registered_signal is not None and signal_reference is not None:
        signal_gain, signal_offset = _gain_offset(
            signal_reference, registered_signal)
        fields['signal_gain'][frame_i] = signal_gain
        fields['signal_offset'][frame_i] = signal_offset
    fields['valid_pixel_fraction'][frame_i] = valid.mean()
    fields['search_boundary'][frame_i] = estimate['search_boundary']
    fields['detector_artifact'][frame_i] = (
        raw_control.min() == raw_control.max() or saturation > 0.01)
    fields['local_reference_fallback'][frame_i] = (
        local_references['canonical_fallback'][local_i])
    fields['_outside_search'][frame_i] = (
        estimate['out_of_range'] or estimate['search_boundary'])
    return registered_control


def _finish_quality(
        fields,
        estimates,
        sampling_frequency_hz,
        *,
        calibration_mask=None,
        focal_mads=2,
        local_references=None,
        ):
    n_frames = len(estimates)
    if calibration_mask is not None:
        calibration_mask = np.asarray(calibration_mask, dtype=bool)
        if calibration_mask.shape != (n_frames,):
            raise ValueError('calibration_mask must have one value per frame')
        if not calibration_mask.any():
            raise ValueError('no usable frames for QC calibration')

    credible_motion = (
        np.isfinite(fields['peak_ratio'])
        & (fields['peak_ratio'] >= 1.1)
        & np.isfinite(fields['tile_disagreement_px'])
        & (fields['tile_disagreement_px'] <= 1.0)
        )
    outside_search = fields.pop('_outside_search')
    baseline = (
        ~fields['detector_artifact']
        & ~fields['timing_fault']
        & ~outside_search
        & credible_motion
        )
    if calibration_mask is not None:
        baseline &= calibration_mask
    # 17 August 2026: two MADs retained every calibration focal frame without
    # ordinary-frame false positives in the controlled-defocus benchmark
    thresholds = _quality_thresholds(fields, baseline, focal_mads)
    if not np.isfinite(list(thresholds.values())).all():
        raise ValueError('no usable frames for QC calibration')
    focal_candidates = _focal_candidates(fields, thresholds)
    fields['threshold_calibration'] = baseline
    change_baseline = (
        baseline
        & np.r_[False, baseline[:-1]]
        & np.r_[baseline[1:], False]
        )
    photometric_baseline = (
        change_baseline
        & (fields['canonical_gradient_ncc'] >= thresholds['canonical_ambiguous'])
        & (fields['local_gradient_ncc'] >= thresholds['local_focal'])
        )
    for field_name, threshold_name, minimum_rise in (
            ('control_gain', 'photometric_control_gain_jump', 0.15),
            ('control_offset', 'photometric_control_offset_jump', 0.4),
            ('signal_gain', 'photometric_signal_gain_jump', 0.15),
            ('signal_offset', 'photometric_signal_offset_jump', 0.4),
            ):
        changes = np.full(n_frames, np.nan, dtype=np.float32)
        changes[1:] = np.abs(np.diff(fields[field_name]))
        fields[f'photometric_{field_name}_change'] = changes
        thresholds[threshold_name] = _high_threshold(
            changes,
            change_baseline,
            minimum_rise,
            6,
            )
    fields['photometric_artifact'] = (
        photometric_baseline
        & (
            (fields['photometric_control_gain_change']
             > thresholds['photometric_control_gain_jump'])
            | (fields['photometric_control_offset_change']
               > thresholds['photometric_control_offset_jump'])
            | (fields['photometric_signal_gain_change']
               > thresholds['photometric_signal_gain_jump'])
            | (fields['photometric_signal_offset_change']
               > thresholds['photometric_signal_offset_jump'])
            )
        )

    for frame_i, estimate in enumerate(estimates):
        low_canonical = (
            fields['canonical_gradient_ncc'][frame_i]
            < thresholds['canonical_focal'])
        low_local = (
            fields['local_gradient_ncc'][frame_i]
            < thresholds['local_focal'])
        reasons = ['timing_fault'] if fields['timing_fault'][frame_i] else []
        motion_reasons = []
        if not np.isfinite(estimate['peak_ratio']) or estimate['peak_ratio'] < 1.1:
            motion_reasons.append('ambiguous_peak')
        if (
                not np.isfinite(estimate['tile_disagreement'])
                or estimate['tile_disagreement'] > 1.0):
            motion_reasons.append('tile_disagreement')
        if fields['detector_artifact'][frame_i]:
            reasons.append('detector_artifact')
            state = 'ambiguous'
        elif outside_search[frame_i]:
            reasons.append(
                'search_boundary' if fields['search_boundary'][frame_i]
                else 'outside_search_range')
            state = 'out_of_range'
        elif fields['photometric_artifact'][frame_i]:
            reasons.append('photometric_artifact')
            state = 'ambiguous'
        elif focal_candidates[frame_i]:
            # 17 August 2026: four measurements retain focal_loss when lateral confidence fails
            reasons.extend(motion_reasons)
            reasons.extend((
                'low_canonical_similarity',
                'low_local_similarity',
                'low_high_frequency_fraction',
                'low_control_gain',
                ))
            state = 'focal_loss'
        else:
            reasons.extend(motion_reasons)
            if low_canonical and low_local:
                reasons.append('loss_of_correspondence')
            elif (
                    fields['canonical_gradient_ncc'][frame_i]
                    < thresholds['canonical_ambiguous']):
                reasons.append('low_canonical_similarity')
            state = 'ambiguous' if reasons else 'accepted'
        fields['recommended_state'][frame_i] = state
        fields['reason_code'][frame_i] = ';'.join(reasons) if reasons else 'accepted'

    fields['analysis_valid'] = fields['recommended_state'] == 'accepted'
    axial_input = fields['canonical_gradient_ncc'].copy()
    axial_input[fields['detector_artifact'] | outside_search] = np.nan
    fields['axial_similarity'] = rolling_axial_similarity(
        axial_input, sampling_frequency_hz, timestamps=fields['time_s'])
    fields['focal_loss_episodes'] = focal_loss_episodes(
        fields['recommended_state'], sampling_frequency_hz,
        timestamps=fields['time_s'], reason_codes=fields['reason_code'])
    fields['local_references'] = local_references
    fields['thresholds'] = {**thresholds, 'focal_mads': float(focal_mads)}
    return fields


def measure_quality(
        reference,
        frames,
        estimates,
        sampling_frequency_hz,
        *,
        calibration_mask=None,
        focal_mads=2,
        local_references=None,
        timestamps=None,
        write_registered=None,
        ):
    n_frames = len(frames)
    if calibration_mask is not None:
        calibration_mask = np.asarray(calibration_mask, dtype=bool)
        if calibration_mask.shape != (n_frames,):
            raise ValueError('calibration_mask must have one value per frame')
        if not calibration_mask.any():
            raise ValueError('no usable frames for QC calibration')
    timestamps = (
        np.arange(n_frames, dtype=float) / sampling_frequency_hz
        if timestamps is None else np.asarray(timestamps, dtype=float)
        )
    if local_references is None:
        local_references = make_local_references(
            reference, frames, estimates, sampling_frequency_hz,
            timestamps=timestamps)
    fields = _new_quality_fields(n_frames, timestamps, sampling_frequency_hz)
    previous_registered = None
    reference_detail = _high_frequency_image(reference)
    reference_gradient = _gradient_image(reference)

    for frame_i, (frame, estimate) in enumerate(zip(frames, estimates)):
        registered, bounds = warp_frame(
            frame, estimate['shift_y'], estimate['shift_x'])
        previous_registered = _measure_registered_quality(
            fields,
            frame_i,
            frame,
            registered,
            estimate,
            reference,
            reference_detail,
            reference_gradient,
            local_references,
            previous_registered,
            )
        if write_registered is not None:
            write_registered(frame_i, registered, bounds)
    return _finish_quality(
        fields,
        estimates,
        sampling_frequency_hz,
        calibration_mask=calibration_mask,
        focal_mads=focal_mads,
        local_references=local_references,
        )


def rolling_axial_similarity(
        similarity,
        sampling_frequency_hz,
        window_s=60,
        *,
        timestamps=None,
        ):
    similarity = np.asarray(similarity, dtype=float)
    timestamps = (
        np.arange(len(similarity), dtype=float) / sampling_frequency_hz
        if timestamps is None else np.asarray(timestamps, dtype=float)
        )
    # 17 August 2026: half-open time windows contain 1,800 observations at 30 Hz
    start = np.searchsorted(timestamps, timestamps - window_s / 2, side='left')
    stop = np.searchsorted(timestamps, timestamps + window_s / 2, side='left')
    finite = np.isfinite(similarity)
    totals = np.r_[0, np.cumsum(np.where(finite, similarity, 0))]
    counts = np.r_[0, np.cumsum(finite)]
    return np.divide(
        totals[stop] - totals[start],
        counts[stop] - counts[start],
        out=np.full(len(similarity), np.nan),
        where=(counts[stop] - counts[start]) > 0,
        ).astype(np.float32)


def focal_loss_episodes(
        states,
        sampling_frequency_hz,
        *,
        timestamps=None,
        reason_codes=None,
        ):
    states = np.asarray(states)
    timestamped = timestamps is not None
    timestamps = (
        np.arange(len(states), dtype=float) / sampling_frequency_hz
        if timestamps is None else np.asarray(timestamps, dtype=float)
        )
    reason_codes = (
        np.full(len(states), 'focal_loss', dtype=object)
        if reason_codes is None else np.asarray(reason_codes)
        )
    frame_period = 1 / sampling_frequency_hz
    if timestamped and len(timestamps) > 1:
        frame_period = float(np.median(np.diff(timestamps)))
    focal = states == 'focal_loss'
    edges = np.diff(np.pad(focal.astype(np.int8), 1))
    starts = list(np.flatnonzero(edges == 1))
    stops = list(np.flatnonzero(edges == -1))
    if not starts:
        return []

    merged = [[starts[0], stops[0]]]
    merge_gap_s = 0.10
    for start, stop in zip(starts[1:], stops[1:]):
        if timestamped:
            previous_stop = merged[-1][1] - 1
            gap_s = timestamps[start] - timestamps[previous_stop] - frame_period
            # 16 August 2026: sampled 0.10 s gaps can sit a few bits above the written boundary
            merge = gap_s <= merge_gap_s or np.isclose(
                gap_s, merge_gap_s, rtol=0, atol=1e-12)
        else:
            merge = start - merged[-1][1] <= merge_gap_s * sampling_frequency_hz
        if merge:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])

    episodes = []
    for start, stop in merged:
        start_time = timestamps[start]
        stop_time = timestamps[stop - 1] + frame_period
        duration = stop_time - start_time
        duration_class = (
            'brief' if duration < 0.25
            else 'transient' if duration <= 0.50
            else 'sustained'
            )
        reasons = set()
        for reason in reason_codes[start:stop][focal[start:stop]]:
            reasons.update(reason.split(';'))
        episodes.append({
            'start_time': float(start_time),
            'stop_time': float(stop_time),
            'duration_s': float(duration),
            'duration_class': duration_class,
            'reason_code': ';'.join(sorted(reasons)),
            'n_frames': int(np.sum(focal[start:stop])),
            })
    return episodes


def add_quality_control_to_nwb(nwbfile, quality):
    module = nwbfile.create_processing_module(
        name='quality_control',
        description='per-frame registration, photometric and focal-loss measurements',
        )
    table = DynamicTable(
        name='registration_qc',
        description='one row for each paired signal/control observation',
        id=np.arange(len(quality['time_s'])),
        )
    # 17 August 2026: dy_px stores shift_y; dx_px stores shift_x
    descriptions = {
        'time_s': 'paired-observation time (s)',
        'dx_px': 'applied control correction along x (px)',
        'dy_px': 'applied control correction along y (px)',
        'displacement_magnitude_px': 'magnitude of the applied control correction (px)',
        'peak_ratio': 'highest to second-highest correlation peak ratio',
        'tile_disagreement_px': 'median local-to-global shift disagreement (px)',
        'canonical_gradient_ncc': 'gradient NCC to the canonical reference (dimensionless)',
        'local_gradient_ncc': 'gradient NCC to the local reference (dimensionless)',
        'high_frequency_fraction': (
            'retained detail between 1.5 and 2.5 px Gaussian scales relative to the reference '
            '(dimensionless)'),
        'spatial_correlation': 'mean horizontal and vertical neighbour correlation',
        'temporal_difference': 'MAD-scaled RMS difference from the previous frame',
        'control_gain': 'control-frame gain fitted against the canonical reference',
        'control_offset': 'control-frame offset fitted against the canonical reference (counts)',
        'signal_gain': 'signal-frame gain fitted against the calibration reference',
        'signal_offset': 'signal-frame offset fitted against the calibration reference (counts)',
        'valid_pixel_fraction': 'fraction remaining after the non-wrapping warp',
        'search_boundary': 'whether the estimate reached the allowed shift boundary',
        'detector_artifact': 'blank or more than 1% of pixels at integer dtype limits',
        'timing_fault': 'timestamp interval differs from the stated period by more than half a frame',
        'photometric_control_gain_change': 'absolute frame-to-frame change in control-frame gain',
        'photometric_control_offset_change': 'absolute frame-to-frame change in control-frame offset',
        'photometric_signal_gain_change': 'absolute frame-to-frame change in signal-frame gain',
        'photometric_signal_offset_change': 'absolute frame-to-frame change in signal-frame offset',
        'photometric_artifact': 'abrupt photometric change detected between frames retaining spatial correspondence',
        'local_reference_fallback': 'local 60 s block used the canonical reference',
        'threshold_calibration': 'frame contributed to the recording-specific QC thresholds',
        'recommended_state': 'accepted, ambiguous, focal_loss or out_of_range',
        'reason_code': 'semicolon-separated reasons for the recommended state',
        'analysis_valid': 'true only for the accepted state',
        }
    for name, description in descriptions.items():
        table.add_column(
            name=name, description=description, data=quality[name])
    module.add(table)

    threshold_descriptions = {
        'canonical_focal': 'focal-loss boundary for canonical_gradient_ncc',
        'local_focal': 'focal-loss boundary for local_gradient_ncc',
        'high_frequency_focal': 'focal-loss boundary for high_frequency_fraction',
        'control_gain_focal': 'focal-loss boundary for control_gain',
        'photometric_control_gain_jump': 'abrupt-change boundary for control_gain',
        'photometric_control_offset_jump': 'abrupt-change boundary for control_offset',
        'photometric_signal_gain_jump': 'abrupt-change boundary for signal_gain',
        'photometric_signal_offset_jump': 'abrupt-change boundary for signal_offset',
        'canonical_ambiguous': 'ambiguous boundary for canonical_gradient_ncc',
        'focal_mads': 'MAD distance used for the four focal-loss boundaries',
        }
    threshold_table = DynamicTable(
        name='registration_thresholds',
        description='thresholds learned from this recording',
        )
    for name, description in threshold_descriptions.items():
        threshold_table.add_column(name=name, description=description)
    threshold_table.add_row(**{
        name: float(quality['thresholds'][name]) for name in threshold_descriptions
        })
    module.add(threshold_table)
    module.add(TimeSeries(
        name='axial_similarity',
        data=quality['axial_similarity'],
        unit='dimensionless',
        timestamps=quality['time_s'],
        description='60 s rolling mean of gradient NCC to the canonical reference',
        ))

    intervals = nwbfile.create_time_intervals(
        name='focal_loss',
        description='focal-loss candidates; no duration is discarded',
        )
    for name, description, dtype in (
            ('duration_s', 'episode duration (s)', np.float64),
            ('duration_class', 'brief, transient or sustained', str),
            ('reason_code', 'semicolon-separated focal-loss reasons', str),
            ('n_frames', 'number of focal-loss frames, excluding merged gaps', np.int64),
            ):
        intervals.add_column(
            name=name, description=description, data=np.asarray([], dtype=dtype))
    for episode in quality['focal_loss_episodes']:
        intervals.add_row(**episode)
    return module


def estimate_channel_offset(signal_frames, control_frames, *, whitening=0):
    signal_frames = np.asarray(signal_frames)
    control_frames = np.asarray(control_frames)
    quarter_shifts = []
    references = []
    for indices in np.array_split(np.arange(len(signal_frames)), 4):
        signal_reference = _mean_registered(signal_frames[indices])
        control_reference = _mean_registered(control_frames[indices])
        signal_reference = np.where(
            np.isfinite(signal_reference), signal_reference, np.nanmedian(signal_reference))
        control_reference = np.where(
            np.isfinite(control_reference), control_reference, np.nanmedian(control_reference))
        estimate = estimate_shift(
            control_reference, signal_reference, whitening=whitening, check_tiles=False)
        quarter_shifts.append([estimate['shift_y'], estimate['shift_x']])
        references.append((signal_reference, control_reference))

    quarter_shifts = np.asarray(quarter_shifts)
    candidate = np.median(quarter_shifts, axis=0)
    disagreement = float(np.max(np.linalg.norm(quarter_shifts - candidate, axis=1)))
    before = np.mean([_gradient_ncc(control, signal) for signal, control in references])
    after = []
    for signal, control in references:
        aligned, _ = warp_frame(signal, *candidate)
        after.append(_gradient_ncc(control, aligned))
    after = float(np.mean(after))
    accepted = disagreement <= 0.25 and after - before >= 0.01
    applied = candidate if accepted else np.zeros(2)
    return {
        'shift_y': float(applied[0]),
        'shift_x': float(applied[1]),
        'candidate_y': float(candidate[0]),
        'candidate_x': float(candidate[1]),
        'quarter_shifts': quarter_shifts,
        'max_disagreement_px': disagreement,
        'gradient_ncc_before': float(before),
        'gradient_ncc_after': after,
        'accepted': bool(accepted),
        }


def _channel_corrections(
        shift_y,
        shift_x,
        signal_to_control_offset,
        registration_channel,
        ):
    dynamic = np.asarray([shift_y, shift_x], dtype=np.float32)
    static = np.asarray(signal_to_control_offset, dtype=np.float32)
    if registration_channel == 'control':
        return dynamic + static, dynamic
    if registration_channel == 'signal':
        return dynamic, dynamic - static
    raise ValueError('registration_channel must be \'signal\' or \'control\'')


def register_pair(
        signal,
        control,
        shift_y,
        shift_x,
        signal_to_control_offset=(0, 0),
        *,
        registration_channel='control',
        ):
    signal_correction, control_correction = _channel_corrections(
        shift_y,
        shift_x,
        signal_to_control_offset,
        registration_channel,
        )
    signal_registered, signal_bounds = warp_frame(signal, *signal_correction)
    control_registered, control_bounds = warp_frame(control, *control_correction)
    return {
        'signal': signal_registered,
        'control': control_registered,
        'signal_bounds': signal_bounds,
        'control_bounds': control_bounds,
        }


def register_pair_piecewise(
        signal,
        control,
        shift_y,
        shift_x,
        field,
        signal_to_control_offset=(0, 0),
        *,
        registration_channel='control',
        focal_loss=False,
        ):
    assessment = assess_tile_field(field, control.shape, focal_loss=focal_loss)
    if assessment['model_used'] == 'rigid':
        result = register_pair(
            signal,
            control,
            shift_y,
            shift_x,
            signal_to_control_offset,
            registration_channel=registration_channel,
            )
        result['model_used'] = 'rigid'
        result['fallback_reason'] = assessment['fallback_reason']
        result['assessment'] = assessment
        return result

    signal_static, control_static = _channel_corrections(
        0,
        0,
        signal_to_control_offset,
        registration_channel,
        )
    signal_registered, signal_bounds, signal_valid = warp_frame_piecewise(
        signal, field, (shift_y, shift_x), signal_static)
    control_registered, control_bounds, control_valid = warp_frame_piecewise(
        control, field, (shift_y, shift_x), control_static)
    return {
        'signal': signal_registered,
        'control': control_registered,
        'signal_bounds': signal_bounds,
        'control_bounds': control_bounds,
        'signal_valid': signal_valid,
        'control_valid': control_valid,
        'model_used': 'piecewise_rigid',
        'fallback_reason': 'accepted',
        'assessment': assessment,
        }


def cross_validated_tile_residuals(tiles, field, shape):
    evidence = (
        {'accepted': tiles['accepted'], 'precision': tiles['precision']}
        if 'accepted' in tiles else _tile_field_evidence(tiles)
        )
    accepted = evidence['accepted']
    rigid_residual = np.hypot(
        tiles['residual_y'][accepted], tiles['residual_x'][accepted])
    piecewise_residual = np.empty(accepted.sum(), dtype=np.float32)
    accepted_index = np.full(len(accepted), -1, dtype=int)
    accepted_index[accepted] = np.arange(accepted.sum())

    _, tile_row = np.unique(tiles['tile_y'], return_inverse=True)
    _, tile_column = np.unique(tiles['tile_x'], return_inverse=True)
    # 18 August 2026: four interleaved folds keep each test tile beside fitted evidence
    folds = 2 * (tile_row % 2) + tile_column % 2
    for fold in range(4):
        test = accepted & (folds == fold)
        training = accepted & (folds != fold)
        training_evidence = {
            'accepted': training,
            'precision': evidence['precision'],
            }
        fitted = fit_tile_field(
            tiles,
            shape,
            int(field['tile_size']),
            spatial_penalty=float(field['spatial_penalty']),
            magnitude_penalty=float(field['magnitude_penalty']),
            evidence=training_evidence,
            )
        difference_y = fitted['predicted_y'][test] - tiles['residual_y'][test]
        difference_x = fitted['predicted_x'][test] - tiles['residual_x'][test]
        piecewise_residual[accepted_index[test]] = np.hypot(
            difference_y, difference_x)
    return rigid_residual, piecewise_residual


def _mean_registered(frames):
    frames = np.asarray(frames)
    finite = np.isfinite(frames)
    total = np.where(finite, frames, 0).sum(axis=0, dtype=np.float64)
    count = finite.sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(frames.shape[1:], np.nan, dtype=np.float64),
        where=count > 0,
        ).astype(np.float32)


def _channel_residual(pairs):
    residual = []
    for indices in np.array_split(np.arange(len(pairs)), 4):
        signal = _mean_registered([pairs[frame_i]['signal'] for frame_i in indices])
        control = _mean_registered([pairs[frame_i]['control'] for frame_i in indices])
        signal = np.where(np.isfinite(signal), signal, np.nanmedian(signal))
        control = np.where(np.isfinite(control), control, np.nanmedian(control))
        estimate = estimate_shift(control, signal, check_tiles=False)
        if not estimate['out_of_range']:
            residual.append(np.hypot(estimate['shift_y'], estimate['shift_x']))
    return float(np.median(residual)) if residual else np.nan


def compare_registration_models(
        reference,
        signal_frames,
        control_frames,
        rigid_estimates,
        tile_results,
        fields,
        *,
        signal_to_control_offset=(0, 0),
        registration_channel='control',
        focal_loss=None,
        ):
    n_frames = len(signal_frames)
    if n_frames < 4:
        raise ValueError('at least four calibration frames are required')
    if not all(len(values) == n_frames for values in (
            control_frames, rigid_estimates, tile_results, fields)):
        raise ValueError('registration calibration inputs have different lengths')
    focal_loss = (
        np.zeros(n_frames, dtype=bool)
        if focal_loss is None else np.asarray(focal_loss, dtype=bool)
        )

    rigid_pairs = []
    piecewise_pairs = []
    rigid_residuals = []
    piecewise_residuals = []
    named_fallback = []
    for frame_i in range(n_frames):
        estimate = rigid_estimates[frame_i]
        rigid = register_pair(
            signal_frames[frame_i],
            control_frames[frame_i],
            estimate['shift_y'],
            estimate['shift_x'],
            signal_to_control_offset,
            registration_channel=registration_channel,
            )
        piecewise = register_pair_piecewise(
            signal_frames[frame_i],
            control_frames[frame_i],
            estimate['shift_y'],
            estimate['shift_x'],
            fields[frame_i],
            signal_to_control_offset,
            registration_channel=registration_channel,
            focal_loss=focal_loss[frame_i],
            )
        rigid_tile_residual, piecewise_tile_residual = (
            cross_validated_tile_residuals(
                tile_results[frame_i], fields[frame_i], reference.shape))
        rigid_residuals.extend(rigid_tile_residual)
        piecewise_residuals.extend(
            rigid_tile_residual
            if piecewise['model_used'] == 'rigid'
            else piecewise_tile_residual)
        rigid_pairs.append(rigid)
        piecewise_pairs.append(piecewise)
        named_fallback.append(piecewise['fallback_reason'] in {
            'accepted',
            'insufficient_tiles',
            'focal_loss',
            'field_overshoot',
            'neighbour_disagreement',
            'jacobian_limit',
            })

    def metrics(pairs, residuals):
        registered = [pair[registration_channel] for pair in pairs]
        valid_fraction = [
            np.mean(np.isfinite(pair['signal']) & np.isfinite(pair['control']))
            for pair in pairs]
        return {
            'gradient_ncc': float(np.nanmean([
                _gradient_ncc(reference, frame) for frame in registered])),
            'residual_p95_px': (
                float(np.percentile(residuals, 95)) if residuals else np.nan),
            'valid_fraction': float(np.mean(valid_fraction)),
            'cross_channel_residual_px': _channel_residual(pairs),
            }

    rigid_metrics = metrics(rigid_pairs, rigid_residuals)
    piecewise_metrics = metrics(piecewise_pairs, piecewise_residuals)
    piecewise_metrics['accepted_or_fallback_fraction'] = float(np.mean(named_fallback))
    decision = select_registration_model(rigid_metrics, piecewise_metrics)
    return {
        **decision,
        'rigid': rigid_metrics,
        'piecewise': piecewise_metrics,
        }


#%% full recording
def _recording_frames(recording, frame_indices):
    return {
        **recording,
        'frames': [recording['frames'][int(frame_i)] for frame_i in frame_indices],
        'n_frames': len(frame_indices),
        }


def _sample_recording(recording, frame_indices, directory, include_signal=False):
    shape = (len(frame_indices), *recording['shape'])
    control = np.lib.format.open_memmap(
        Path(directory) / 'control_sample.npy', mode='w+',
        dtype=recording['dtype'], shape=shape)
    signal = (
        np.lib.format.open_memmap(
            Path(directory) / 'signal_sample.npy', mode='w+',
            dtype=recording['dtype'], shape=shape)
        if include_signal else None)
    for sample_i, pair in enumerate(read_tiffs(
            _recording_frames(recording, frame_indices))):
        if signal is not None:
            signal[sample_i] = pair['signal']
        control[sample_i] = pair['control']
    if signal is not None:
        signal.flush()
    control.flush()
    return signal, control


def _load_calibration_frames(recording, frame_indices):
    shape = (len(frame_indices), *recording['shape'])
    signal = np.empty(shape, dtype=recording['dtype'])
    control = np.empty(shape, dtype=recording['dtype'])
    for frame_i, pair in enumerate(read_tiffs(
            _recording_frames(recording, frame_indices))):
        signal[frame_i] = pair['signal']
        control[frame_i] = pair['control']
    return signal, control


def _registration_calibration(
        recording,
        reference,
        registration_model,
        registration_channel,
        ):
    calibration_indices = np.linspace(
        0, max(3, recording['n_frames'] // 2 - 1),
        min(100, max(4, recording['n_frames'] // 2)), dtype=int)
    signal_frames, control_frames = _load_calibration_frames(
        recording, calibration_indices)
    motion_frames = (
        control_frames if registration_channel == 'control' else signal_frames)
    estimates = _estimate_shifts(_prepare_reference(reference), motion_frames)
    aligned = [
        register_pair(
            signal, control, estimate['shift_y'], estimate['shift_x'],
            registration_channel=registration_channel)
        for signal, control, estimate in zip(
            signal_frames, control_frames, estimates)
        ]
    channel_offset = estimate_channel_offset(
        [pair['signal'] for pair in aligned],
        [pair['control'] for pair in aligned],
        )
    signal_to_control_offset = (
        channel_offset['shift_y'], channel_offset['shift_x'])
    calibration_accepted = np.asarray([
        estimate['peak_ratio'] >= 1.1 and not estimate['out_of_range']
        for estimate in estimates], dtype=bool)
    signal_shifts = [
        _channel_corrections(
            estimate['shift_y'],
            estimate['shift_x'],
            signal_to_control_offset,
            registration_channel,
            )[0]
        for estimate in estimates]
    signal_reference = _registered_mean(
        signal_frames, signal_shifts, calibration_accepted, recording['shape'])

    if registration_model == 'rigid':
        return {
            'selected_model': 'rigid',
            'requested_model': registration_model,
            'calibration_indices': calibration_indices,
            'channel_offset': channel_offset,
            'signal_reference': signal_reference,
            'model_comparison': None,
            'tile_reference': None,
            }

    # 19 August 2026: 80 px and penalty 10 are the settings carried by the
    # ten-source piecewise benchmark into the production path
    tile_size = min(80, min(reference.shape))
    tile_reference = _prepare_tile_reference(reference, tile_size)
    tile_results = []
    fields = []
    for frame, estimate in zip(motion_frames, estimates):
        tiles = _estimate_tile_shifts(
            tile_reference,
            frame,
            (estimate['shift_y'], estimate['shift_x']),
            )
        evidence = _tile_field_evidence(tiles)
        field = fit_tile_field(
            tiles,
            reference.shape,
            tile_size,
            spatial_penalty=10,
            magnitude_penalty=1,
            evidence=evidence,
            )
        tile_results.append(tiles)
        fields.append(field)
    comparison = compare_registration_models(
        reference,
        signal_frames,
        control_frames,
        estimates,
        tile_results,
        fields,
        signal_to_control_offset=signal_to_control_offset,
        registration_channel=registration_channel,
        )
    selected_model = (
        comparison['selected_model']
        if registration_model == 'auto' else 'piecewise_rigid')
    return {
        'selected_model': selected_model,
        'requested_model': registration_model,
        'calibration_indices': calibration_indices,
        'channel_offset': channel_offset,
        'signal_reference': signal_reference,
        'model_comparison': comparison,
        'tile_reference': tile_reference,
        }


def _control_local_references(
        control_reference,
        registration_reference,
        signal_sample,
        control_sample,
        sample_indices,
        timestamps,
        sampling_frequency_hz,
        registration_channel,
        channel_offset,
        ):
    motion_sample = (
        control_sample if registration_channel == 'control' else signal_sample)
    estimates = _estimate_shifts(
        _prepare_reference(registration_reference), motion_sample)
    signal_to_control_offset = (
        channel_offset['shift_y'], channel_offset['shift_x'])
    control_estimates = []
    for estimate in estimates:
        _, control_correction = _channel_corrections(
            estimate['shift_y'],
            estimate['shift_x'],
            signal_to_control_offset,
            registration_channel,
            )
        control_estimates.append({
            **estimate,
            'shift_y': float(control_correction[0]),
            'shift_x': float(control_correction[1]),
            })
    sample_times = sample_indices / sampling_frequency_hz
    sampled = make_local_references(
        control_reference,
        control_sample,
        control_estimates,
        sampling_frequency_hz,
        timestamps=sample_times,
        )
    full_window = np.floor(
        (timestamps - sample_times[0]) / 60).astype(np.int64)
    # uniform reference sampling gives every one-minute interval at least one frame
    reference_index = np.searchsorted(sampled['window'], full_window)
    return {
        **sampled,
        'reference_index': reference_index.astype(np.int32),
        }


def _add_recording_tables(
        module,
        recording,
        session_time_source,
        pixel_size_um,
        calibration,
        ):
    metadata = DynamicTable(
        name='recording_metadata',
        description='recording-level values used for preprocessing',
        )
    columns = {
        'sampling_frequency_hz': 'paired observations per second',
        'multiplexed': 'whether signal and control pages alternate in one TIFF',
        'signal_channel': 'one-based signal position within each TIFF pair',
        'control_channel': 'one-based control position within each TIFF pair',
        'signal_label': 'signal-channel label',
        'control_label': 'control-channel label',
        'pixel_size_um': 'pixel width and height in micrometres; NaN when unknown',
        'source_dtype': 'dtype of the raw TIFF pages',
        'session_start_time_source': 'source used for the required NWB session time',
        'session_start_time_raw': 'unparsed source value',
        'session_start_time_timezone': 'timezone applied to the source value',
        }
    for name, description in columns.items():
        metadata.add_column(name=name, description=description)
    metadata.add_row(
        sampling_frequency_hz=recording['sampling_frequency_hz'],
        multiplexed=recording['multiplexed'],
        signal_channel=recording['signal_channel'],
        control_channel=recording['control_channel'],
        signal_label=recording['signal_label'] or '',
        control_label=recording['control_label'] or '',
        pixel_size_um=np.nan if pixel_size_um is None else pixel_size_um,
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
    source_tiffs.add_column(name='sha256', description='SHA-256 digest of the TIFF bytes')
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
        digest = hashlib.sha256()
        with path.open('rb') as file:
            while block := file.read(1024 ** 2):
                digest.update(block)
        source_tiffs.add_row(
            path=str(path.resolve()),
            channel_role='/'.join(roles),
            n_pages=n_pages,
            sha256=digest.hexdigest(),
            )
    module.add(source_tiffs)

    frame_table = DynamicTable(
        name='recording_index',
        description='TIFF pages for each paired frame',
        id=np.arange(recording['n_frames']),
        )
    frame_columns = {
        'frame': ('zero-based paired frame in the source recording', [
            frame['frame'] for frame in recording['frames']]),
        'signal_tiff': ('absolute path of the signal TIFF', [
            str(frame['signal_tiff'].resolve()) for frame in recording['frames']]),
        'signal_page': ('zero-based signal page within that TIFF', [
            frame['signal_page'] for frame in recording['frames']]),
        'control_tiff': ('absolute path of the control TIFF', [
            str(frame['control_tiff'].resolve()) for frame in recording['frames']]),
        'control_page': ('zero-based control page within that TIFF', [
            frame['control_page'] for frame in recording['frames']]),
        }
    for name, (description, data) in frame_columns.items():
        frame_table.add_column(name=name, description=description, data=data)
    module.add(frame_table)

    offset = calibration['channel_offset']
    channel_alignment = DynamicTable(
        name='channel_alignment',
        description='static signal-to-control translation',
        )
    for name, description in (
            ('dy_px', 'applied signal-to-control correction along y'),
            ('dx_px', 'applied signal-to-control correction along x'),
            ('candidate_dy_px', 'median candidate correction along y'),
            ('candidate_dx_px', 'median candidate correction along x'),
            ('max_disagreement_px', 'largest quartile disagreement from the candidate'),
            ('gradient_ncc_before', 'mean gradient NCC before alignment'),
            ('gradient_ncc_after', 'mean gradient NCC after alignment'),
            ('accepted', 'whether the candidate met both acceptance boundaries'),
            ):
        channel_alignment.add_column(name=name, description=description)
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

    model = DynamicTable(
        name='registration_model',
        description='requested model and calibration decision',
        )
    for name, description in (
            ('requested_model', 'rigid, piecewise or auto'),
            ('selected_model', 'model used for the recording'),
            ('calibration_frames', 'number of time-balanced calibration frames'),
            ('comparison', 'JSON-encoded candidate metrics, thresholds and decisions'),
            ):
        model.add_column(name=name, description=description)
    comparison = calibration['model_comparison']
    model.add_row(
        requested_model=calibration['requested_model'],
        selected_model=calibration['selected_model'],
        calibration_frames=len(calibration['calibration_indices']),
        comparison=(
            '' if comparison is None else json.dumps(
                comparison, default=lambda value: value.item())),
        )
    module.add(model)


def _new_preprocessing_file(
        partial_path,
        recording,
        session_start_time,
        session_time_source,
        pixel_size_um,
        control_reference,
        registration_reference,
        local_references,
        calibration,
        registration_channel,
        ):
    n_frames = recording['n_frames']
    height, width = recording['shape']
    timestamps = np.arange(n_frames, dtype=float) / recording['sampling_frequency_hz']
    nwbfile = NWBFile(
        session_description='FibreSight preprocessed recording',
        identifier=f'fibre-sight-{partial_path.stem}-{session_start_time.isoformat()}',
        session_start_time=session_start_time,
        timestamps_reference_time=session_start_time,
        )
    module = nwbfile.create_processing_module(
        name='preprocessing',
        description='registered movies and the measurements used to create them',
        )
    paired_frames = TimeSeries(
        name='paired_frames',
        data=np.arange(n_frames, dtype=np.int64),
        unit='frame',
        timestamps=timestamps,
        description='source frame number and paired-observation time',
        )
    module.add(paired_frames)
    # 19 August 2026: NWB image axes are x, y; TIFF and NumPy images are row-y, column-x
    for channel in ('signal', 'control'):
        empty = np.empty((0, width, height), dtype=np.int16)
        module.add(ImageSeries(
            name=f'registered_{channel}',
            data=H5DataIO(
                empty,
                chunks=(1, width, height),
                maxshape=(None, width, height),
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
            description=f'registered {channel} movie; zero denotes pixels outside the valid area',
            ))

    images = [GrayscaleImage(
        name='control_reference',
        data=control_reference.T,
        description='two-pass control-channel reference',
        )]
    if registration_channel == 'signal':
        images.append(GrayscaleImage(
            name='signal_reference',
            data=registration_reference.T,
            description='two-pass signal-channel registration reference',
            ))
    images.extend(GrayscaleImage(
        name=f'local_control_reference_{reference_i:03d}',
        data=reference.T,
        description='control reference for one 60 s interval',
        ) for reference_i, reference in enumerate(local_references['images']))
    module.add(Images(
        name='registration_references',
        images=images,
        description='canonical and local images used for registration QC',
        ))
    _add_recording_tables(
        module,
        recording,
        session_time_source,
        pixel_size_um,
        calibration,
        )

    with NWBHDF5IO(partial_path, 'w') as io:
        io.write(nwbfile)
    with h5py.File(partial_path, 'r+') as file:
        for channel in ('signal', 'control'):
            file[f'processing/preprocessing/registered_{channel}/data'].resize(
                (n_frames, width, height))


def _int16_registered(frame):
    valid = np.isfinite(frame)
    rounded = np.rint(frame[valid])
    limits = np.iinfo(np.int16)
    if np.any((rounded < limits.min) | (rounded > limits.max)):
        raise ValueError('registered intensity exceeds the int16 storage range')
    stored = np.zeros(frame.shape, dtype=np.int16)
    stored[valid] = rounded.astype(np.int16)
    return stored


def _calculate_segmentation_references(
        nwb_path,
        percentiles=SEGMENTATION_REFERENCE_PERCENTILES,
        ):
    low_percentile, high_percentile = map(float, percentiles)
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError('segmentation percentiles must satisfy 0 <= low < high <= 100')

    with h5py.File(nwb_path, 'r') as file:
        reference_path = 'processing/preprocessing/segmentation_references'
        if reference_path in file:
            raise ValueError('segmentation reference already exists')

        analysis_valid = np.asarray(
            file['processing/quality_control/registration_qc/analysis_valid'],
            dtype=bool,
            )
        frame_indices = np.flatnonzero(analysis_valid)
        if not len(frame_indices):
            raise ValueError('no analysis-valid frames remain for segmentation reference')

        movie = file['processing/preprocessing/registered_control/data']
        bounds = file['processing/preprocessing/registered_valid_bounds']
        shape = (movie.shape[2], movie.shape[1])
        total = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.uint32)
        for frame_i in frame_indices:
            y0 = int(bounds['control_y0'][frame_i])
            y1 = int(bounds['control_y1'][frame_i])
            x0 = int(bounds['control_x0'][frame_i])
            x1 = int(bounds['control_x1'][frame_i])
            frame = np.asarray(movie[frame_i]).T
            total[y0:y1, x0:x1] += frame[y0:y1, x0:x1]
            count[y0:y1, x0:x1] += 1

    mean_image = np.divide(
        total,
        count,
        out=np.zeros_like(total),
        where=count > 0,
        ).astype(np.float32)
    low_value, high_value = np.percentile(
        mean_image, [low_percentile, high_percentile])
    if high_value == low_value:
        raise ValueError('segmentation reference percentile range is zero')
    segmentation_image = np.clip(
        (mean_image - low_value) / (high_value - low_value),
        0,
        1,
        )
    segmentation_image = (segmentation_image * 255).astype(np.uint8)
    return {
        'mean_image': mean_image,
        'segmentation_image': segmentation_image,
        'frame_indices': frame_indices.astype(np.int64),
        'low_percentile': low_percentile,
        'high_percentile': high_percentile,
        'low_value': float(low_value),
        'high_value': float(high_value),
        }


def _write_segmentation_references(nwb_path, references):
    with NWBHDF5IO(nwb_path, 'a') as io:
        nwbfile = io.read()
        module = nwbfile.processing['preprocessing']
        if 'segmentation_references' in module.data_interfaces:
            raise ValueError('segmentation reference already exists')
        module.add(Images(
            name='segmentation_references',
            description=(
                'registered control-channel references used to construct and '
                'run ROI segmentation'),
            images=[
                GrayscaleImage(
                    name='mean_control_reference',
                    data=references['mean_image'].T,
                    description=(
                        f'mean of all {len(references["frame_indices"]):,} '
                        'analysis-valid registered control frames'),
                    ),
                GrayscaleImage(
                    name='segmentation_reference',
                    data=references['segmentation_image'].T,
                    description=(
                        '8-bit percentile-clipped mean control reference used '
                        'for ROI inference'),
                    ),
                ],
            ))
        frames = DynamicTable(
            name='segmentation_reference_frames',
            description='registered frames averaged into the mean control reference',
            id=np.arange(len(references['frame_indices'])),
            )
        frames.add_column(
            name='frame_index',
            description='zero-based paired-frame index',
            data=references['frame_indices'],
            )
        module.add(frames)
        metadata = DynamicTable(
            name='segmentation_reference_metadata',
            description='provenance for stored segmentation inference references',
            )
        columns = {
            'reference_path': 'NWB path of the inference reference',
            'source_reference_path': 'NWB path of the raw mean used as its source',
            'frame_count': 'number of analysis-valid registered frames averaged',
            'low_percentile': 'lower percentile used for clipping and rescaling',
            'high_percentile': 'upper percentile used for clipping and rescaling',
            'low_value': 'source-image intensity at the lower percentile',
            'high_value': 'source-image intensity at the upper percentile',
            'output_dtype': 'stored inference-reference dtype',
            }
        for name, description in columns.items():
            metadata.add_column(name=name, description=description)
        metadata.add_row(
            reference_path=(
                'processing/preprocessing/segmentation_references/'
                'segmentation_reference'),
            source_reference_path=(
                'processing/preprocessing/segmentation_references/'
                'mean_control_reference'),
            frame_count=len(references['frame_indices']),
            low_percentile=references['low_percentile'],
            high_percentile=references['high_percentile'],
            low_value=references['low_value'],
            high_value=references['high_value'],
            output_dtype='uint8',
            )
        module.add(metadata)
        io.write(nwbfile)


def _verify_segmentation_references(nwb_path, references):
    with NWBHDF5IO(nwb_path, 'r') as io:
        module = io.read().processing['preprocessing']
        stored_mean = np.asarray(
            module['segmentation_references']['mean_control_reference'].data).T
        stored_segmentation = np.asarray(
            module['segmentation_references']['segmentation_reference'].data).T
        frame_indices = np.asarray(
            module['segmentation_reference_frames']['frame_index'], dtype=np.int64)
        metadata = module['segmentation_reference_metadata']
        stored_metadata = {
            name: metadata[name][0]
            for name in metadata.colnames
            }
    np.testing.assert_array_equal(stored_mean, references['mean_image'])
    np.testing.assert_array_equal(
        stored_segmentation, references['segmentation_image'])
    np.testing.assert_array_equal(frame_indices, references['frame_indices'])
    if int(stored_metadata['frame_count']) != len(references['frame_indices']):
        raise AssertionError('stored segmentation reference frame count changed')
    for name in ('low_percentile', 'high_percentile', 'low_value', 'high_value'):
        if float(stored_metadata[name]) != references[name]:
            raise AssertionError(f'stored segmentation reference {name} changed')


def _append_segmentation_references(
        nwb_path,
        percentiles=SEGMENTATION_REFERENCE_PERCENTILES,
        ):
    references = _calculate_segmentation_references(nwb_path, percentiles)
    _write_segmentation_references(nwb_path, references)
    _verify_segmentation_references(nwb_path, references)
    return references


def add_segmentation_references(
        nwb_path,
        percentiles=SEGMENTATION_REFERENCE_PERCENTILES,
        ):
    nwb_path = Path(nwb_path)
    partial_path = nwb_path.with_name(
        f'{nwb_path.stem}.segmentation-reference.partial.nwb')
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')

    references = _calculate_segmentation_references(nwb_path, percentiles)
    original_size = nwb_path.stat().st_size
    copy_start = perf_counter()
    shutil.copyfile(nwb_path, partial_path)
    copy_time_s = perf_counter() - copy_start
    _write_segmentation_references(partial_path, references)

    validation_errors = [str(error) for error in validate(path=partial_path)]
    if validation_errors:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    _verify_segmentation_references(partial_path, references)
    with partial_path.open('rb') as file:
        os.fsync(file.fileno())
    file_size_bytes = partial_path.stat().st_size
    os.replace(partial_path, nwb_path)
    return {
        'output_path': nwb_path,
        'frame_count': len(references['frame_indices']),
        'percentiles': (
            references['low_percentile'], references['high_percentile']),
        'copy_time_s': copy_time_s,
        'file_size_bytes': file_size_bytes,
        'file_size_increase_bytes': file_size_bytes - original_size,
        'validation_errors': validation_errors,
        }


def _add_full_registration(
        partial_path,
        recording,
        estimates,
        quality,
        signal_bounds,
        control_bounds,
        piecewise_results,
        ):
    with NWBHDF5IO(partial_path, 'a') as io:
        nwbfile = io.read()
        module = nwbfile.processing['preprocessing']
        paired_frames = module['paired_frames']
        translations = np.asarray([
            [estimate['shift_y'], estimate['shift_x']]
            for estimate in estimates
            ], dtype=np.float32)
        module.add(TimeSeries(
            name='rigid_translation',
            data=translations,
            unit='pixels',
            timestamps=paired_frames,
            description='applied registration-channel translation in dy_px, dx_px order',
            ))

        bounds = DynamicTable(
            name='registered_valid_bounds',
            description=(
                'enclosing half-open y/x rectangles after the stored x/y image is returned '
                'to row/column matrix order; rigid bounds are exact and piecewise edges follow '
                'the stored spline field'),
            id=np.arange(recording['n_frames']),
            )
        for channel, channel_bounds in (
                ('signal', signal_bounds), ('control', control_bounds)):
            for coordinate_i, coordinate in enumerate(('y0', 'y1', 'x0', 'x1')):
                bounds.add_column(
                    name=f'{channel}_{coordinate}',
                    description=f'{channel} valid {coordinate} (px)',
                    data=channel_bounds[:, coordinate_i],
                    )
        module.add(bounds)

        if piecewise_results is not None:
            table = DynamicTable(
                name='piecewise_registration',
                description='per-frame field acceptance and rigid fallback',
                id=np.arange(recording['n_frames']),
                )
            descriptions = {
                'model_used': 'piecewise_rigid or rigid',
                'fallback_reason': 'accepted or named field rejection reason',
                'global_shift_y_px': 'field-wide local adjustment along y',
                'global_shift_x_px': 'field-wide local adjustment along x',
                'accepted_tile_fraction': 'fraction of tiles retained for the field fit',
                'field_rms_px': 'RMS local displacement in pixels',
                'field_max_px': 'maximum local displacement in pixels',
                'neighbour_difference_max_px': 'maximum adjacent-tile difference in pixels',
                'jacobian_min': 'minimum field Jacobian determinant',
                'jacobian_max': 'maximum field Jacobian determinant',
                }
            for name, description in descriptions.items():
                table.add_column(
                    name=name,
                    description=description,
                    data=piecewise_results[name],
                    )
            module.add(table)
            module.add(TimeSeries(
                name='piecewise_spline_coefficients',
                data=np.stack([
                    piecewise_results['coefficient_y'],
                    piecewise_results['coefficient_x'],
                    ], axis=1),
                unit='pixels',
                timestamps=paired_frames,
                description='cubic B-spline coefficients in y, x order',
                ))
            spline_grid = DynamicTable(
                name='piecewise_spline_grid',
                description='control-point coordinates for the stored spline coefficients',
                )
            for name, description in (
                    ('y_index', 'coefficient-array y index'),
                    ('x_index', 'coefficient-array x index'),
                    ('y_px', 'control-point row'),
                    ('x_px', 'control-point column'),
                    ):
                spline_grid.add_column(name=name, description=description)
            for y_i, y in enumerate(piecewise_results['control_y']):
                for x_i, x in enumerate(piecewise_results['control_x']):
                    spline_grid.add_row(y_index=y_i, x_index=x_i, y_px=y, x_px=x)
            module.add(spline_grid)
            module.add(TimeSeries(
                name='piecewise_accepted_tiles',
                data=piecewise_results['accepted_tiles'],
                unit='boolean',
                timestamps=paired_frames,
                description='tile acceptance mask used for each field fit',
                ))
            module.add(TimeSeries(
                name='piecewise_tile_peak_ratio',
                data=piecewise_results['tile_peak_ratio'],
                unit='dimensionless',
                timestamps=paired_frames,
                description='local highest to second-highest peak ratio',
                ))
            grid = DynamicTable(
                name='piecewise_tile_grid',
                description='fixed tile centres used for local translation estimates',
                )
            grid.add_column(name='y_px', description='tile-centre row')
            grid.add_column(name='x_px', description='tile-centre column')
            for y, x in zip(
                    piecewise_results['tile_y'], piecewise_results['tile_x']):
                grid.add_row(y_px=y, x_px=x)
            module.add(grid)
        add_quality_control_to_nwb(nwbfile, quality)
        io.write(nwbfile)


def _spot_check_preprocessed(path, expected_frames, recording):
    with NWBHDF5IO(path, 'r') as io:
        nwbfile = io.read()
        module = nwbfile.processing['preprocessing']
        sampling_frequency = module['recording_metadata']['sampling_frequency_hz'][0]
        if sampling_frequency != recording['sampling_frequency_hz']:
            raise AssertionError('stored sampling frequency changed')
        for frame_i, expected in expected_frames.items():
            for channel in ('signal', 'control'):
                stored = np.asarray(
                    module[f'registered_{channel}'].data[frame_i]).T
                if not np.array_equal(stored, expected[channel]):
                    raise AssertionError(
                        f'{channel} frame {frame_i} changed after NWB writing')
    return nwbfile.session_start_time


def preprocess_recording(
        tiff_paths,
        output_path,
        signal_channel,
        control_channel,
        multiplexed,
        sampling_frequency_hz,
        *,
        signal_label=None,
        control_label=None,
        control_tiff_paths=None,
        registration_model='auto',
        registration_channel='control',
        pixel_size_um=None,
        session_start_time=None,
        segmentation_reference_percentiles=SEGMENTATION_REFERENCE_PERCENTILES,
        ):
    output_path = Path(output_path)
    partial_path = output_path.with_suffix('.partial.nwb')
    if output_path.exists():
        raise FileExistsError(f'output already exists: {output_path}')
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')
    if registration_model not in {'rigid', 'piecewise', 'auto'}:
        raise ValueError('registration_model must be rigid, piecewise or auto')
    if registration_channel not in {'signal', 'control'}:
        raise ValueError('registration_channel must be signal or control')

    start = perf_counter()
    recording = index_tiffs(
        tiff_paths,
        signal_channel=signal_channel,
        control_channel=control_channel,
        sampling_frequency_hz=sampling_frequency_hz,
        multiplexed=multiplexed,
        control_tiffs=control_tiff_paths,
        signal_label=signal_label,
        control_label=control_label,
        )
    if session_start_time is None:
        session_start_time, session_time_source = read_session_start_time(
            recording['signal_tiffs'][0])
    else:
        session_time_source = {
            'source': 'user-supplied',
            'raw_value': session_start_time.isoformat(),
            'timezone': str(session_start_time.tzinfo),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.linspace(
        0, recording['n_frames'] - 1,
        min(1000, recording['n_frames']), dtype=int)
    with tempfile.TemporaryDirectory(
            prefix='fibre sight reference ', dir=output_path.parent) as sample_dir:
        signal_sample, control_sample = _sample_recording(
            recording,
            sample_indices,
            sample_dir,
            include_signal=registration_channel == 'signal',
            )
        control_reference_info = make_reference(control_sample)
        control_reference = control_reference_info['image']
        if registration_channel == 'signal':
            registration_reference = make_reference(signal_sample)['image']
        else:
            registration_reference = control_reference
        calibration = _registration_calibration(
            recording,
            registration_reference,
            registration_model,
            registration_channel,
            )
        timestamps = (
            np.arange(recording['n_frames'], dtype=float)
            / recording['sampling_frequency_hz'])
        local_references = _control_local_references(
            control_reference,
            registration_reference,
            signal_sample,
            control_sample,
            sample_indices,
            timestamps,
            recording['sampling_frequency_hz'],
            registration_channel,
            calibration['channel_offset'],
            )

        _new_preprocessing_file(
            partial_path,
            recording,
            session_start_time,
            session_time_source,
            pixel_size_um,
            control_reference,
            registration_reference,
            local_references,
            calibration,
            registration_channel,
            )

        n_frames = recording['n_frames']
        estimates = []
        signal_bounds = np.empty((n_frames, 4), dtype=np.int32)
        control_bounds = np.empty((n_frames, 4), dtype=np.int32)
        quality_fields = _new_quality_fields(
            n_frames, timestamps, recording['sampling_frequency_hz'])
        # fixed reference derivatives are shared by every frame and both NCC measurements
        reference_detail = _high_frequency_image(control_reference)
        reference_gradient = _gradient_image(control_reference)
        previous_control = None
        prepared_reference = _prepare_reference(registration_reference)
        selected_model = calibration['selected_model']
        tile_reference = calibration['tile_reference']
        signal_to_control_offset = (
            calibration['channel_offset']['shift_y'],
            calibration['channel_offset']['shift_x'])
        piecewise_results = None
        if selected_model == 'piecewise_rigid':
            n_tiles = len(tile_reference['tile_y'])
            control_y, control_x = _spline_grid(
                recording['shape'], tile_reference['tile_size'])
            piecewise_results = {
                'model_used': np.empty(n_frames, dtype='<U16'),
                'fallback_reason': np.empty(n_frames, dtype='<U24'),
                'global_shift_y_px': np.empty(n_frames, dtype=np.float32),
                'global_shift_x_px': np.empty(n_frames, dtype=np.float32),
                'accepted_tile_fraction': np.empty(n_frames, dtype=np.float32),
                'field_rms_px': np.empty(n_frames, dtype=np.float32),
                'field_max_px': np.empty(n_frames, dtype=np.float32),
                'neighbour_difference_max_px': np.empty(n_frames, dtype=np.float32),
                'jacobian_min': np.empty(n_frames, dtype=np.float32),
                'jacobian_max': np.empty(n_frames, dtype=np.float32),
                'coefficient_y': np.empty(
                    (n_frames, len(control_y), len(control_x)), dtype=np.float32),
                'coefficient_x': np.empty(
                    (n_frames, len(control_y), len(control_x)), dtype=np.float32),
                'accepted_tiles': np.empty((n_frames, n_tiles), dtype=bool),
                'tile_peak_ratio': np.empty((n_frames, n_tiles), dtype=np.float32),
                'tile_y': tile_reference['tile_y'],
                'tile_x': tile_reference['tile_x'],
                'control_y': control_y,
                'control_x': control_x,
                }

        spot_indices = set(np.random.default_rng(42).choice(
            n_frames, min(5, n_frames), replace=False)) | {0, n_frames - 1}
        expected_frames = {}
        # 32 frames amortise HDF5 writes and hold about 32 MiB for both channels
        frame_batch_size = 32
        estimate_frame = partial(_estimate_shift, prepared_reference)
        with h5py.File(partial_path, 'r+') as file:
            signal_data = file['processing/preprocessing/registered_signal/data']
            control_data = file['processing/preprocessing/registered_control/data']
            frame_i = 0
            frame_iter = iter(read_tiffs(recording))
            # NumPy, SciPy and OpenCV release the GIL here; four workers match the benchmark ceiling
            with ThreadPoolExecutor(max_workers=4) as pool:
                while pairs := list(islice(frame_iter, frame_batch_size)):
                    estimates_batch = list(pool.map(
                        estimate_frame,
                        [pair[registration_channel] for pair in pairs]))
                    estimates.extend(estimates_batch)
                    batch_start = frame_i

                    if selected_model == 'piecewise_rigid':
                        registered_batch = []
                        for batch_i, (pair, estimate) in enumerate(
                                zip(pairs, estimates_batch)):
                            result_i = batch_start + batch_i
                            tiles = _estimate_tile_shifts(
                                tile_reference,
                                pair[registration_channel],
                                (estimate['shift_y'], estimate['shift_x']),
                                )
                            evidence = _tile_field_evidence(tiles)
                            field = fit_tile_field(
                                tiles,
                                recording['shape'],
                                tile_reference['tile_size'],
                                spatial_penalty=10,
                                magnitude_penalty=1,
                                evidence=evidence,
                                )
                            registered = register_pair_piecewise(
                                pair['signal'],
                                pair['control'],
                                estimate['shift_y'],
                                estimate['shift_x'],
                                field,
                                signal_to_control_offset,
                                registration_channel=registration_channel,
                                )
                            registered_batch.append(registered)
                            assessment = registered['assessment']
                            for name in (
                                    'model_used', 'fallback_reason',
                                    'accepted_tile_fraction', 'field_rms_px',
                                    'field_max_px', 'neighbour_difference_max_px',
                                    'jacobian_min', 'jacobian_max'):
                                piecewise_results[name][result_i] = assessment[name]
                            use_field = registered['model_used'] == 'piecewise_rigid'
                            piecewise_results['global_shift_y_px'][result_i] = (
                                field['global_shift_y'] if use_field else 0)
                            piecewise_results['global_shift_x_px'][result_i] = (
                                field['global_shift_x'] if use_field else 0)
                            piecewise_results['coefficient_y'][result_i] = (
                                field['coefficient_y'] if use_field else 0)
                            piecewise_results['coefficient_x'][result_i] = (
                                field['coefficient_x'] if use_field else 0)
                            piecewise_results['accepted_tiles'][result_i] = field['accepted']
                            piecewise_results['tile_peak_ratio'][result_i] = tiles['peak_ratio']
                    else:
                        registration_jobs = [pool.submit(
                            register_pair,
                            pair['signal'],
                            pair['control'],
                            estimate['shift_y'],
                            estimate['shift_x'],
                            signal_to_control_offset,
                            registration_channel=registration_channel,
                            ) for pair, estimate in zip(pairs, estimates_batch)]
                        registered_batch = [job.result() for job in registration_jobs]

                    stored_signal_batch = np.empty(
                        (len(pairs), *recording['shape']), dtype=np.int16)
                    stored_control_batch = np.empty_like(stored_signal_batch)
                    quality_jobs = []
                    previous_for_quality = previous_control
                    for batch_i, (pair, estimate, registered) in enumerate(
                            zip(pairs, estimates_batch, registered_batch)):
                        result_i = batch_start + batch_i
                        stored_signal = _int16_registered(registered['signal'])
                        stored_control = _int16_registered(registered['control'])
                        stored_signal_batch[batch_i] = stored_signal
                        stored_control_batch[batch_i] = stored_control
                        signal_bounds[result_i] = registered['signal_bounds']
                        control_bounds[result_i] = registered['control_bounds']
                        quality_jobs.append(pool.submit(
                            _measure_registered_quality,
                            quality_fields,
                            result_i,
                            pair['control'],
                            registered['control'],
                            estimate,
                            control_reference,
                            reference_detail,
                            reference_gradient,
                            local_references,
                            previous_for_quality,
                            registered_signal=registered['signal'],
                            signal_reference=calibration['signal_reference'],
                            ))
                        previous_for_quality = registered['control']
                        if result_i in spot_indices:
                            expected_frames[result_i] = {
                                'signal': stored_signal.copy(),
                                'control': stored_control.copy(),
                                }
                    frame_i += len(pairs)
                    signal_data[batch_start:frame_i] = stored_signal_batch.swapaxes(1, 2)
                    control_data[batch_start:frame_i] = stored_control_batch.swapaxes(1, 2)
                    for job in quality_jobs:
                        job.result()
                    previous_control = previous_for_quality
            file.flush()

        quality = _finish_quality(
            quality_fields,
            estimates,
            recording['sampling_frequency_hz'],
            local_references=local_references,
            )
        _add_full_registration(
            partial_path,
            recording,
            estimates,
            quality,
            signal_bounds,
            control_bounds,
            piecewise_results,
            )
        segmentation_references = _append_segmentation_references(
            partial_path,
            segmentation_reference_percentiles,
            )

    validation_errors = [str(error) for error in validate(path=partial_path)]
    if validation_errors:
        raise AssertionError(f'NWB validation failed: {validation_errors}')
    _spot_check_preprocessed(partial_path, expected_frames, recording)
    with partial_path.open('rb') as file:
        os.fsync(file.fileno())
    os.replace(partial_path, output_path)
    return {
        'output_path': output_path,
        'n_frames': recording['n_frames'],
        'selected_model': calibration['selected_model'],
        'focal_loss_episodes': len(quality['focal_loss_episodes']),
        'validation_errors': validation_errors,
        'wall_time_s': perf_counter() - start,
        'file_size_bytes': output_path.stat().st_size,
        'session_start_time': session_start_time,
        'session_time_source': session_time_source,
        'reference_fallback': bool(control_reference_info['reference_fallback']),
        'segmentation_reference_frames': len(
            segmentation_references['frame_indices']),
        'segmentation_reference_percentiles': (
            segmentation_references['low_percentile'],
            segmentation_references['high_percentile'],
            ),
        }

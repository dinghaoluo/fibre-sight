'''
Created on 14 August 2026
Modified on 18 August 2026

read paired TIFF frames, correct rigid and piecewise movement, and record registration QC

@author: Dinghao Luo
'''

#%% imports
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from pathlib import Path
import re

import cv2
import numpy as np
from scipy import fft, ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.signal.windows import tukey
import tifffile


#%% TIFF metadata
_NUMBER = re.compile(r'(\d+)')


def _sort_tiffs(paths):
    def sort_key(path):
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in _NUMBER.split(path.name)
            )

    return sorted((Path(path) for path in paths), key=sort_key)


def _description_value(description, name):
    match = re.search(rf'^{re.escape(name)}\s*=\s*(.+?)\s*$', description, re.MULTILINE)
    return match.group(1) if match else None


def _saved_channels(page):
    software = page.tags.get('Software')
    if software is None:
        return None
    value = _description_value(str(software.value), 'SI.hChannels.channelSave')
    if value is None:
        return None
    return tuple(int(number) for number in re.findall(r'\d+', value))


def _frame_info(page):
    description = page.description or ''
    frame = _description_value(description, 'frameNumbers')
    time_s = _description_value(description, 'frameTimestamps_sec')
    return (
        int(frame) if frame is not None else None,
        float(time_s) if time_s is not None else None,
        )


#%% page pairing
def _multiplexed_pairs(tiff_files, signal_channel, control_channel):
    channel_order = None
    for path in tiff_files:
        with tifffile.TiffFile(path) as tiff:
            channels = _saved_channels(tiff.pages[0])
            if channels is None:
                raise ValueError(f'missing ScanImage channelSave metadata: {path}')
            if len(channels) != 2 or set(channels) != {signal_channel, control_channel}:
                raise ValueError(
                    f'saved channels {channels} do not match '
                    f'{signal_channel}/{control_channel}: {path}'
                    )
            if channel_order is None:
                channel_order = channels
            elif channels != channel_order:
                raise ValueError(f'saved channel order changes from {channel_order} to {channels}')
            if len(tiff.pages) % 2:
                raise ValueError(f'incomplete signal/control pair: {path}')

            signal_offset = channels.index(signal_channel)
            control_offset = channels.index(control_channel)
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


#%% reader
def read_tiffs(
        signal_tiffs,
        *,
        signal_channel,
        control_channel,
        sampling_frequency_hz,
        multiplexed=True,
        control_tiffs=None,
        ):
    if signal_channel < 1 or control_channel < 1 or signal_channel == control_channel:
        raise ValueError('signal and control channels must be different one-based numbers')
    if sampling_frequency_hz <= 0:
        raise ValueError('sampling_frequency_hz must be positive')

    signal_tiffs = _sort_tiffs(signal_tiffs)
    if not signal_tiffs:
        raise ValueError('no signal TIFFs found')

    if multiplexed:
        pairs = _multiplexed_pairs(signal_tiffs, signal_channel, control_channel)
    else:
        if control_tiffs is None:
            raise ValueError('control_tiffs is required for separate TIFFs')
        control_tiffs = _sort_tiffs(control_tiffs)
        pairs = _separate_pairs(signal_tiffs, control_tiffs)

    shape = None
    dtype = None
    prev_time = None
    for frame, pair in enumerate(pairs):
        signal_page, control_page, signal_path, control_path, signal_i, control_i = pair
        if signal_page.shape != control_page.shape or signal_page.dtype != control_page.dtype:
            raise ValueError('paired signal and control frames have different shape or dtype')
        if shape is None:
            shape = signal_page.shape
            dtype = signal_page.dtype
        elif signal_page.shape != shape or signal_page.dtype != dtype:
            raise ValueError('TIFF frame shape or dtype changes within the recording')

        signal_frame, signal_time = _frame_info(signal_page)
        control_frame, control_time = _frame_info(control_page)
        if (
                signal_frame is not None
                and control_frame is not None
                and signal_frame != control_frame
                ):
            raise ValueError('paired ScanImage frames have different frame numbers')
        if (
                signal_time is not None
                and control_time is not None
                and signal_time != control_time
                ):
            raise ValueError('paired ScanImage frames have different timestamps')

        time_s = signal_time if signal_time is not None else control_time
        if time_s is None:
            time_s = 0.0 if prev_time is None else prev_time + 1 / sampling_frequency_hz
        if prev_time is not None:
            if time_s <= prev_time:
                raise ValueError('TIFF timestamps do not increase')

        yield {
            'frame': frame,
            'time_s': time_s,
            'signal': signal_page.asarray(),
            'control': control_page.asarray(),
            'signal_tiff': signal_path,
            'control_tiff': control_path,
            'signal_page': signal_i,
            'control_page': control_i,
        }
        prev_time = time_s


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
    source_y, source_x, valid, bounds = field_coordinates(
        field, frame.shape, rigid_shift, static_offset)
    registered = cv2.remap(
        np.asarray(frame, dtype=np.float32),
        source_x,
        source_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
        )
    registered[~valid] = np.nan
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


def _gradient_ncc(first, second):
    first = np.hypot(ndi.sobel(first, axis=0), ndi.sobel(first, axis=1))
    second = np.hypot(ndi.sobel(second, axis=0), ndi.sobel(second, axis=1))
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    power = np.sqrt(np.dot(first, first) * np.dot(second, second))
    return float(np.dot(first, second) / power) if power else np.nan


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
    interior = ndi.binary_erosion(valid, iterations=8)
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
        'reference_index': reference_index,
        'canonical_fallback': np.asarray(canonical_fallback, dtype=bool),
        }


def _temporal_difference(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    first = first[valid]
    second = second[valid]
    first_mad = np.median(np.abs(first - np.median(first)))
    second_mad = np.median(np.abs(second - np.median(second)))
    if not first_mad or not second_mad:
        return np.nan
    first = (first - np.median(first)) / first_mad
    second = (second - np.median(second)) / second_mad
    return float(np.sqrt(np.mean((first - second) ** 2)))


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
    frame_period = 1 / sampling_frequency_hz
    timing_fault = np.zeros(n_frames, dtype=bool)
    timing_fault[1:] = np.abs(np.diff(timestamps) - frame_period) > frame_period / 2
    fields = {
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
        'valid_pixel_fraction': np.empty(n_frames, dtype=np.float32),
        'search_boundary': np.empty(n_frames, dtype=bool),
        'detector_artifact': np.empty(n_frames, dtype=bool),
        'timing_fault': timing_fault,
        'local_reference_fallback': np.empty(n_frames, dtype=bool),
        'recommended_state': np.empty(n_frames, dtype=object),
        'reason_code': np.empty(n_frames, dtype=object),
    }
    outside_search = np.empty(n_frames, dtype=bool)
    previous_registered = None
    reference_detail = _high_frequency_image(reference)

    for frame_i, (frame, estimate) in enumerate(zip(frames, estimates)):
        registered, bounds = warp_frame(
            frame, estimate['shift_y'], estimate['shift_x'])
        local_i = local_references['reference_index'][frame_i]
        local_reference = local_references['images'][local_i]
        canonical_ncc = _gradient_ncc(reference, registered)
        local_ncc = _gradient_ncc(local_reference, registered)
        high_frequency = _high_frequency_fraction(reference_detail, registered)
        spatial_correlation = _spatial_correlation(registered)
        gain, offset = _gain_offset(reference, registered)
        y0, y1, x0, x1 = bounds
        valid_fraction = (y1 - y0) * (x1 - x0) / frame.size
        limits = np.iinfo(frame.dtype) if np.issubdtype(frame.dtype, np.integer) else None
        saturation = (
            np.mean((frame == limits.min) | (frame == limits.max))
            if limits is not None else 0
            )
        detector_artifact = frame.min() == frame.max() or saturation > 0.01
        search_boundary = estimate['search_boundary']

        fields['dx_px'][frame_i] = estimate['shift_x']
        fields['dy_px'][frame_i] = estimate['shift_y']
        fields['displacement_magnitude_px'][frame_i] = np.hypot(
            estimate['shift_y'], estimate['shift_x'])
        fields['peak_ratio'][frame_i] = estimate['peak_ratio']
        fields['tile_disagreement_px'][frame_i] = estimate['tile_disagreement']
        fields['canonical_gradient_ncc'][frame_i] = canonical_ncc
        fields['local_gradient_ncc'][frame_i] = local_ncc
        fields['high_frequency_fraction'][frame_i] = high_frequency
        fields['spatial_correlation'][frame_i] = spatial_correlation
        if previous_registered is not None:
            fields['temporal_difference'][frame_i] = _temporal_difference(
                previous_registered, registered)
        fields['control_gain'][frame_i] = gain
        fields['control_offset'][frame_i] = offset
        fields['valid_pixel_fraction'][frame_i] = valid_fraction
        fields['search_boundary'][frame_i] = search_boundary
        fields['detector_artifact'][frame_i] = detector_artifact
        fields['local_reference_fallback'][frame_i] = (
            local_references['canonical_fallback'][local_i])
        outside_search[frame_i] = estimate['out_of_range'] or search_boundary
        if write_registered is not None:
            write_registered(frame_i, registered, bounds)
        previous_registered = registered

    credible_motion = (
        np.isfinite(fields['peak_ratio'])
        & (fields['peak_ratio'] >= 1.1)
        & np.isfinite(fields['tile_disagreement_px'])
        & (fields['tile_disagreement_px'] <= 1.0)
        )
    baseline = (
        ~fields['detector_artifact']
        & ~fields['timing_fault']
        & ~outside_search
        & credible_motion
        )
    if calibration_mask is not None:
        baseline &= calibration_mask
    # 17 August 2026: two MADs was the strongest controlled-defocus boundary which
    # kept all 153 calibration focal frames and gave no ordinary-frame false positives
    thresholds = _quality_thresholds(fields, baseline, focal_mads)
    if not np.isfinite(list(thresholds.values())).all():
        raise ValueError('no usable frames for QC calibration')
    focal_candidates = _focal_candidates(fields, thresholds)
    fields['threshold_calibration'] = baseline

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
        elif focal_candidates[frame_i]:
            # 17 August 2026: four focal measurements retain focal_loss when lateral confidence fails
            reasons.extend(motion_reasons)
            reasons.append('low_canonical_similarity')
            reasons.extend((
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
        axial_input, sampling_frequency_hz, timestamps=timestamps)
    fields['focal_loss_episodes'] = focal_loss_episodes(
        fields['recommended_state'], sampling_frequency_hz,
        timestamps=timestamps, reason_codes=fields['reason_code'])
    fields['local_references'] = local_references
    fields['thresholds'] = {**thresholds, 'focal_mads': float(focal_mads)}
    return fields


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
    from hdmf.common import DynamicTable
    from pynwb import TimeSeries

    module = nwbfile.create_processing_module(
        name='quality_control',
        description='per-frame registration and focal-loss measurements',
        )
    table = DynamicTable(
        name='rigid_registration_qc',
        description='one row for each paired signal/control observation',
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
        'valid_pixel_fraction': 'fraction remaining after the non-wrapping warp',
        'search_boundary': 'whether the estimate reached the allowed shift boundary',
        'detector_artifact': 'blank or more than 1% of pixels at integer dtype limits',
        'timing_fault': 'timestamp interval differs from the stated period by more than half a frame',
        'local_reference_fallback': 'local 60 s block used the canonical reference',
        'threshold_calibration': 'frame contributed to the recording-specific QC thresholds',
        'recommended_state': 'accepted, ambiguous, focal_loss or out_of_range',
        'reason_code': 'semicolon-separated reasons for the recommended state',
        'analysis_valid': 'true only for the accepted state',
        }
    for name, description in descriptions.items():
        table.add_column(name=name, description=description)
    for frame_i in range(len(quality['time_s'])):
        table.add_row(**{
            name: quality[name][frame_i].item()
            if isinstance(quality[name][frame_i], np.generic)
            else quality[name][frame_i]
            for name in descriptions
            })
    module.add(table)

    threshold_descriptions = {
        'canonical_focal': 'focal-loss boundary for canonical_gradient_ncc',
        'local_focal': 'focal-loss boundary for local_gradient_ncc',
        'high_frequency_focal': 'focal-loss boundary for high_frequency_fraction',
        'control_gain_focal': 'focal-loss boundary for control_gain',
        'canonical_ambiguous': 'ambiguous boundary for canonical_gradient_ncc',
        'focal_mads': 'MAD distance used for the four focal-loss boundaries',
        }
    threshold_table = DynamicTable(
        name='rigid_registration_thresholds',
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
        signal_reference = np.nanmean(signal_frames[indices], axis=0)
        control_reference = np.nanmean(control_frames[indices], axis=0)
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


def register_pair(signal, control, shift_y, shift_x, signal_offset=(0, 0)):
    control_registered, control_bounds = warp_frame(control, shift_y, shift_x)
    signal_shift = shift_y + signal_offset[0], shift_x + signal_offset[1]
    signal_registered, signal_bounds = warp_frame(signal, *signal_shift)
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
        assessment,
        signal_offset=(0, 0),
        ):
    if assessment['model_used'] == 'rigid':
        result = register_pair(
            signal, control, shift_y, shift_x, signal_offset=signal_offset)
        result['model_used'] = 'rigid'
        result['fallback_reason'] = assessment['fallback_reason']
        return result

    control_registered, control_bounds, control_valid = warp_frame_piecewise(
        control, field, (shift_y, shift_x))
    signal_registered, signal_bounds, signal_valid = warp_frame_piecewise(
        signal, field, (shift_y, shift_x), signal_offset)
    return {
        'signal': signal_registered,
        'control': control_registered,
        'signal_bounds': signal_bounds,
        'control_bounds': control_bounds,
        'signal_valid': signal_valid,
        'control_valid': control_valid,
        'model_used': 'piecewise_rigid',
        'fallback_reason': 'accepted',
        }

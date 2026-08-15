'''
Created on 14 August 2026

read paired TIFF frames and correct rigid movement

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import re

import numpy as np
from scipy import ndimage as ndi
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
def _registration_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = image - np.median(image)
    scale = np.median(np.abs(image))
    if scale == 0:
        return np.zeros_like(image)

    image = ndi.gaussian_filter(image / scale, 1)
    image -= ndi.gaussian_filter(image, 21)
    window = np.outer(tukey(image.shape[0], 0.2), tukey(image.shape[1], 0.2))
    return image * window


def _local_dft(data, region_size, upsample, offsets):
    properties = list(zip(data.shape, [region_size] * data.ndim, offsets))
    for n_items, size, offset in properties[::-1]:
        frequency = np.fft.fftfreq(n_items, d=upsample)
        kernel = np.exp(
            -2j * np.pi * (np.arange(size) - offset)[:, None] * frequency[None, :]
            )
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _shift_from_images(reference, frame, max_shift, upsample, whitening):
    reference_fft = np.fft.fftn(reference)
    frame_fft = np.fft.fftn(frame)
    product = reference_fft * frame_fft.conj()
    if whitening:
        magnitude = np.abs(product)
        product /= np.maximum(magnitude, np.finfo(magnitude.dtype).eps) ** whitening

    correlation = np.abs(np.fft.ifftn(product))
    y = np.arange(reference.shape[0])
    x = np.arange(reference.shape[1])
    y[y > reference.shape[0] // 2] -= reference.shape[0]
    x[x > reference.shape[1] // 2] -= reference.shape[1]
    yy, xx = np.meshgrid(y, x, indexing='ij')
    allowed = (np.abs(yy) <= max_shift[0]) & (np.abs(xx) <= max_shift[1])

    global_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = np.unravel_index(np.argmax(np.where(allowed, correlation, -np.inf)), correlation.shape)
    shift = np.array([yy[peak], xx[peak]], dtype=float)
    out_of_range = not allowed[global_peak]

    away_from_peak = (np.abs(yy - shift[0]) > 2) | (np.abs(xx - shift[1]) > 2)
    second_peak = correlation[allowed & away_from_peak].max()
    peak_ratio = correlation[peak] / second_peak

    region_size = int(np.ceil(upsample * 1.5))
    centre = np.trunc(region_size / 2)
    offsets = centre - shift * upsample
    local = _local_dft(product.conj(), region_size, upsample, offsets).conj()
    local_peak = np.unravel_index(np.argmax(np.abs(local)), local.shape)
    shift += (np.asarray(local_peak) - centre) / upsample
    out_of_range |= np.any(np.abs(shift) > np.asarray(max_shift) + 1 / upsample)
    return shift, float(peak_ratio), bool(out_of_range)


def estimate_shift(
        reference,
        frame,
        *,
        max_shift_fraction=0.1,
        upsample=10,
        whitening=0,
        check_tiles=True,
        ):
    '''estimate the translation applied to align one frame with a reference'''
    reference = np.asarray(reference)
    frame = np.asarray(frame)
    max_shift = np.maximum(1, np.round(np.asarray(reference.shape) * max_shift_fraction)).astype(int)
    shift, peak_ratio, out_of_range = _shift_from_images(
        _registration_image(reference),
        _registration_image(frame),
        max_shift,
        upsample,
        whitening,
        )

    tile_disagreement = np.nan
    if check_tiles:
        tile_shape = np.maximum(32, np.round(np.asarray(reference.shape) * 0.6)).astype(int)
        tile_shifts = []
        for y in (0, reference.shape[0] - tile_shape[0]):
            for x in (0, reference.shape[1] - tile_shape[1]):
                tile = np.s_[y:y + tile_shape[0], x:x + tile_shape[1]]
                tile_shift, _, _ = _shift_from_images(
                    _registration_image(reference[tile]),
                    _registration_image(frame[tile]),
                    max_shift,
                    upsample,
                    whitening,
                    )
                tile_shifts.append(tile_shift)
        tile_shifts = np.asarray(tile_shifts)
        tile_disagreement = float(np.median(np.linalg.norm(tile_shifts - shift, axis=1)))

    return {
        'shift_y': float(shift[0]),
        'shift_x': float(shift[1]),
        'peak_ratio': peak_ratio,
        'tile_disagreement': tile_disagreement,
        'out_of_range': out_of_range,
        }


def _valid_bounds(shape, shift_y, shift_x):
    y0 = max(0, int(np.ceil(shift_y)))
    y1 = min(shape[0], int(np.floor(shape[0] - 1 + shift_y)) + 1)
    x0 = max(0, int(np.ceil(shift_x)))
    x1 = min(shape[1], int(np.floor(shape[1] - 1 + shift_x)) + 1)
    return y0, y1, x0, x1


def warp_frame(frame, shift_y, shift_x):
    '''apply one bilinear translation and return its valid bounds'''
    registered = ndi.shift(
        np.asarray(frame, dtype=np.float32),
        (shift_y, shift_x),
        order=1,
        mode='constant',
        cval=np.nan,
        prefilter=False,
        )
    return registered, _valid_bounds(frame.shape, shift_y, shift_x)


def _registered_mean(frames, shifts, accepted):
    total = np.zeros(frames.shape[1:], dtype=np.float64)
    count = np.zeros(frames.shape[1:], dtype=np.uint16)
    for frame, shift, use_frame in zip(frames, shifts, accepted):
        if not use_frame:
            continue
        registered, _ = warp_frame(frame, *shift)
        valid = np.isfinite(registered)
        total[valid] += registered[valid]
        count[valid] += 1
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
    '''make a two-pass reference from evenly sampled frames'''
    n_frames = len(frames)
    if n_frames < min_frames:
        raise ValueError(f'at least {min_frames} frames are required for the reference')

    sample_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    sample = np.stack([np.asarray(frames[int(i)]) for i in sample_indices])
    median = np.empty(len(sample), dtype=np.float32)
    mad = np.empty(len(sample), dtype=np.float32)
    gradient = np.empty(len(sample), dtype=np.float32)
    saturation = np.zeros(len(sample), dtype=np.float32)
    limits = np.iinfo(sample.dtype) if np.issubdtype(sample.dtype, np.integer) else None
    for frame_i, frame in enumerate(sample):
        image = np.asarray(frame, dtype=np.float32)
        median[frame_i] = np.median(image)
        mad[frame_i] = np.median(np.abs(image - median[frame_i]))
        gradient[frame_i] = np.mean(
            np.hypot(ndi.sobel(image, axis=0), ndi.sobel(image, axis=1)))
        if limits is not None:
            saturation[frame_i] = np.mean((frame == limits.min) | (frame == limits.max))

    informative = (mad > 0) & (saturation <= 0.01)
    if informative.sum() < min_frames:
        raise ValueError(f'fewer than {min_frames} informative frames remain')
    informative &= gradient >= 0.1 * np.median(gradient[informative])
    if informative.sum() < min_frames:
        raise ValueError(f'fewer than {min_frames} informative frames remain')

    candidates = np.flatnonzero(informative)
    anchor_candidates = candidates[
        np.linspace(0, len(candidates) - 1, min(100, len(candidates)), dtype=int)
        ]
    small = np.stack([_registration_image(sample[i])[::4, ::4].ravel() for i in anchor_candidates])
    small -= small.mean(axis=1, keepdims=True)
    small /= np.linalg.norm(small, axis=1, keepdims=True)
    similarity = small @ small.T
    anchor = anchor_candidates[np.argmax(np.median(similarity, axis=1))]

    first_shifts = np.full((len(sample), 2), np.nan, dtype=np.float32)
    first_accepted = np.zeros(len(sample), dtype=bool)
    for frame_i in candidates:
        estimate = estimate_shift(
            sample[anchor], sample[frame_i], whitening=whitening, check_tiles=False)
        first_shifts[frame_i] = estimate['shift_y'], estimate['shift_x']
        first_accepted[frame_i] = (
            estimate['peak_ratio'] >= min_peak_ratio and not estimate['out_of_range']
            )
    if first_accepted.sum() < min_frames:
        raise ValueError(f'fewer than {min_frames} frames align with the first reference')
    provisional = _registered_mean(sample, first_shifts, first_accepted)

    shifts = np.full((len(sample), 2), np.nan, dtype=np.float32)
    peak_ratio = np.full(len(sample), np.nan, dtype=np.float32)
    tile_disagreement = np.full(len(sample), np.nan, dtype=np.float32)
    accepted = np.zeros(len(sample), dtype=bool)
    for frame_i in candidates:
        estimate = estimate_shift(provisional, sample[frame_i], whitening=whitening)
        shifts[frame_i] = estimate['shift_y'], estimate['shift_x']
        peak_ratio[frame_i] = estimate['peak_ratio']
        tile_disagreement[frame_i] = estimate['tile_disagreement']
        accepted[frame_i] = (
            estimate['peak_ratio'] >= min_peak_ratio
            and estimate['tile_disagreement'] <= max_tile_disagreement
            and not estimate['out_of_range']
            )
    if accepted.sum() < min_frames:
        raise ValueError(f'fewer than {min_frames} frames align with the provisional reference')

    # 14 August 2026: the synthetic reference error began to flatten at 500 frames
    reference = _registered_mean(sample, shifts, accepted)
    return {
        'image': reference,
        'sample_indices': sample_indices,
        'accepted': accepted,
        'shift_y': shifts[:, 0],
        'shift_x': shifts[:, 1],
        'peak_ratio': peak_ratio,
        'tile_disagreement': tile_disagreement,
        'gradient_information': gradient,
        'saturation_fraction': saturation,
        }


def _gradient_ncc(first, second):
    first = np.hypot(ndi.sobel(first, axis=0), ndi.sobel(first, axis=1))
    second = np.hypot(ndi.sobel(second, axis=0), ndi.sobel(second, axis=1))
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    return float(np.dot(first, second) / np.sqrt(np.dot(first, first) * np.dot(second, second)))


def estimate_channel_offset(signal_frames, control_frames, *, whitening=0):
    '''estimate the fixed translation which places signal in control coordinates'''
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
    '''apply shared movement and the fixed signal-to-control translation once'''
    control_registered, control_bounds = warp_frame(control, shift_y, shift_x)
    signal_shift = shift_y + signal_offset[0], shift_x + signal_offset[1]
    signal_registered, signal_bounds = warp_frame(signal, *signal_shift)
    return {
        'signal': signal_registered,
        'control': control_registered,
        'signal_bounds': signal_bounds,
        'control_bounds': control_bounds,
        }

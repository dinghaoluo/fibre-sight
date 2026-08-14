'''
Created on 14 August 2026

read paired signal and control frames from TIFF stacks

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import re

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

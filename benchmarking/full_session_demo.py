'''
Created on 21 August 2026

summarise the private full-session demonstration run in one public figure

@author: Dinghao Luo
'''

#%% imports
import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from pynwb import NWBHDF5IO
from scipy import ndimage as ndi
import tifffile

from fibre_sight._formatting import mpl_formatting
from fibre_sight.api import load_dff_run, load_fluorescence_run, load_roi_run
from fibre_sight.roi_io import roi_dict_to_label


#%% paths and analysis choices
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / 'examples' / 'full_session_demo_workflow.png'

ROI_RUN = 'bundled_proposals_v3'
FLUORESCENCE_RUN = 'full_session_fluorescence_v3'
DFF_RUN = 'full_session_dff_v4'
STIMULATION_THRESHOLD = 100
MAXIMUM_STIMULATION_GAP_FRAMES = 1
ALIGNMENT_START_S = -1.0
ALIGNMENT_STOP_S = 3.0
LOCAL_BASELINE_START_S = -1.0
LOCAL_BASELINE_STOP_S = -0.6
RESPONSE_START_S = 0.1
RESPONSE_STOP_S = 1.0
RAW_MEAN_FRAMES = 30

SIGNAL_COLOUR = '#009E73'
CONTROL_COLOUR = '#D55E00'
NEUTRAL_COLOUR = '#4D4D4D'
AMBIGUOUS_COLOUR = '#E69F00'
OUT_OF_RANGE_COLOUR = '#CC79A7'


#%% small calculations
def _read_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _normalise_image(image):
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / (high - low), 0, 1)


def _finite_mean(values, axis, keepdims=False):
    values = np.asarray(values)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis, keepdims=keepdims)
    total = np.sum(np.where(valid, values, 0), axis=axis, keepdims=keepdims)
    mean = np.full(np.shape(total), np.nan, dtype=np.float32)
    np.divide(total, count, out=mean, where=count > 0)
    return mean


def _finite_sem(values, axis):
    values = np.asarray(values)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    mean = _finite_mean(values, axis=axis, keepdims=True)
    squared_error = np.sum(
        np.where(valid, (values - mean) ** 2, 0), axis=axis)
    variance_of_mean = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(
        squared_error,
        count * (count - 1),
        out=variance_of_mean,
        where=count > 1,
        )
    return np.sqrt(variance_of_mean)


def _merged_runs(mask, maximum_gap_frames):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    runs = []
    for start, stop in zip(starts, stops):
        if runs and start - runs[-1][1] <= maximum_gap_frames:
            runs[-1] = (runs[-1][0], int(stop))
        else:
            runs.append((int(start), int(stop)))
    return runs


def _align(values, off_frames, before_frames, after_frames):
    return np.stack([
        values[off_frame - before_frames:off_frame + after_frames + 1]
        for off_frame in off_frames
        ])


def _event_analysis(fluorescence, dff, sampling_frequency_hz):
    traces = fluorescence['traces']
    raw_signal = np.nanmedian(traces['signal_roi_mean'], axis=1)
    raw_control = np.nanmedian(traces['control_roi_mean'], axis=1)
    pulses = _merged_runs(
        raw_control > STIMULATION_THRESHOLD,
        MAXIMUM_STIMULATION_GAP_FRAMES,
        )
    off_frames = np.asarray([stop for _, stop in pulses], dtype=np.int64)

    before_frames = int(round(-ALIGNMENT_START_S * sampling_frequency_hz))
    after_frames = int(round(ALIGNMENT_STOP_S * sampling_frequency_hz))
    time_s = np.arange(-before_frames, after_frames + 1) / sampling_frequency_hz
    baseline_window = (
        (time_s >= LOCAL_BASELINE_START_S)
        & (time_s <= LOCAL_BASELINE_STOP_S)
        )
    response_window = (
        (time_s >= RESPONSE_START_S)
        & (time_s <= RESPONSE_STOP_S)
        )

    aligned = {}
    trial_means = {}
    responses = {}
    population = {}
    for channel in ('signal', 'control'):
        values = dff['traces'][f'{channel}_roi_dff']
        channel_aligned = _align(
            values, off_frames, before_frames, after_frames)
        local_baseline = _finite_mean(
            channel_aligned[:, baseline_window], axis=1, keepdims=True)
        channel_aligned = channel_aligned - local_baseline
        trial_mean = _finite_mean(channel_aligned, axis=0)
        response = _finite_mean(trial_mean[response_window], axis=0)
        pulse_population = np.ma.median(
            np.ma.masked_invalid(channel_aligned), axis=2).filled(np.nan)
        population_mean = _finite_mean(pulse_population, axis=0)
        population_sem = _finite_sem(pulse_population, axis=0)
        aligned[channel] = channel_aligned
        trial_means[channel] = trial_mean
        responses[channel] = response
        population[channel] = (population_mean, population_sem)

    return {
        'raw_signal': raw_signal,
        'raw_control': raw_control,
        'pulses': pulses,
        'off_frames': off_frames,
        'time_s': time_s,
        'aligned': aligned,
        'trial_means': trial_means,
        'responses': responses,
        'population': population,
        }


#%% NWB context
def _read_nwb_context(nwb_path):
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        preprocessing = nwbfile.processing['preprocessing']
        qc = nwbfile.processing['quality_control']['registration_qc']
        metadata = preprocessing['recording_metadata']
        recording_index = preprocessing['recording_index']

        timestamps = np.asarray(preprocessing['paired_frames'].timestamps).copy()
        signal_paths = [_read_text(value) for value in recording_index['signal_tiff'][:]]
        control_paths = [_read_text(value) for value in recording_index['control_tiff'][:]]
        first_signal_path = Path(signal_paths[0])
        first_control_path = Path(control_paths[0])
        signal_pages = np.asarray(
            recording_index['signal_page'][:RAW_MEAN_FRAMES], dtype=int)
        control_pages = np.asarray(
            recording_index['control_page'][:RAW_MEAN_FRAMES], dtype=int)
        context = {
            'timestamps': timestamps,
            'sampling_frequency_hz': float(metadata['sampling_frequency_hz'][0]),
            'analysis_valid': np.asarray(qc['analysis_valid'], dtype=bool).copy(),
            'displacement_magnitude_px': np.asarray(
                qc['displacement_magnitude_px'], dtype=float).copy(),
            'recommended_state': np.asarray([
                _read_text(value) for value in qc['recommended_state'][:]
                ]),
            }

    if first_signal_path == first_control_path:
        with tifffile.TiffFile(first_signal_path) as tiff:
            context['raw_signal'] = np.mean(
                [tiff.pages[index].asarray() for index in signal_pages],
                axis=0,
                dtype=np.float32,
                )
            context['raw_control'] = np.mean(
                [tiff.pages[index].asarray() for index in control_pages],
                axis=0,
                dtype=np.float32,
                )
    else:
        with tifffile.TiffFile(first_signal_path) as tiff:
            context['raw_signal'] = np.mean(
                [tiff.pages[index].asarray() for index in signal_pages],
                axis=0,
                dtype=np.float32,
                )
        with tifffile.TiffFile(first_control_path) as tiff:
            context['raw_control'] = np.mean(
                [tiff.pages[index].asarray() for index in control_pages],
                axis=0,
                dtype=np.float32,
                )
    return context


#%% figure parts
def _roi_overlay(reference, roi_dict):
    image = _normalise_image(reference)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    labelled, _ = roi_dict_to_label(roi_dict, reference.shape)
    mask = labelled > 0
    boundary = mask & ~ndi.binary_erosion(mask)
    colour = np.asarray(mcolors.to_rgb(SIGNAL_COLOUR))
    rgb[mask] = 0.72 * rgb[mask] + 0.28 * colour
    rgb[boundary] = colour
    return rgb


def _image_axis(axis, image, title):
    axis.imshow(_normalise_image(image), cmap='gray', interpolation='nearest')
    axis.set_title(title, loc='left', fontsize=9)
    axis.set_axis_off()


def _style_time_axis(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', color='0.9', linewidth=0.6)
    axis.tick_params(length=3)


#%% full figure
def plot_full_session_demo(nwb_path, out_path):
    nwb_path = Path(nwb_path)
    out_path = Path(out_path)
    context = _read_nwb_context(nwb_path)
    rois = load_roi_run(nwb_path, ROI_RUN)
    fluorescence = load_fluorescence_run(nwb_path, FLUORESCENCE_RUN)
    dff = load_dff_run(nwb_path, DFF_RUN)
    events = _event_analysis(
        fluorescence,
        dff,
        context['sampling_frequency_hz'],
        )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mpl_formatting()
    plt.rcParams.update({
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'legend.fontsize': 8,
        })

    figure = plt.figure(figsize=(15, 8))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.975,
        bottom=0.085,
        top=0.88,
        hspace=0.46,
        wspace=0.27,
        )

    raw_grid = grid[0, 0].subgridspec(1, 2, wspace=0.04)
    signal_axis = figure.add_subplot(raw_grid[0, 0])
    control_axis = figure.add_subplot(raw_grid[0, 1])
    _image_axis(signal_axis, context['raw_signal'], 'Raw dLight (1 s mean)')
    _image_axis(control_axis, context['raw_control'], 'Raw tdTomato (1 s mean)')

    roi_axis = figure.add_subplot(grid[1, 0])
    roi_axis.imshow(
        _roi_overlay(rois['reference'], rois['roi_dict']),
        interpolation='nearest',
        )
    reference_provenance = rois['reference_provenance']
    reference_frame_count = int(reference_provenance['frame_count'])
    low_percentile = float(reference_provenance['low_percentile'])
    high_percentile = float(reference_provenance['high_percentile'])
    roi_axis.set_title(
        f'tdTomato proposals ({reference_frame_count:,} frames, '
        f'p{low_percentile:g}-p{high_percentile:g})',
        loc='left',
        )
    roi_axis.set_axis_off()

    timestamps = context['timestamps']
    fluorescence_axis = figure.add_subplot(grid[1, 1])
    fluorescence_axis.plot(
        timestamps,
        events['raw_signal'],
        color=SIGNAL_COLOUR,
        linewidth=0.7,
        label='dLight',
        )
    fluorescence_axis.plot(
        timestamps,
        events['raw_control'],
        color=CONTROL_COLOUR,
        linewidth=0.7,
        label='tdTomato',
        )
    fluorescence_axis.axhline(
        STIMULATION_THRESHOLD,
        color=CONTROL_COLOUR,
        linewidth=0.8,
        linestyle=':',
        label='pulse threshold',
        )
    fluorescence_axis.set_yscale('symlog', linthresh=1)
    fluorescence_axis.set_title('Full-session ROI fluorescence', loc='left')
    fluorescence_axis.set_xlabel('session time (s)')
    fluorescence_axis.set_ylabel('median ROI fluorescence (counts)')
    fluorescence_axis.legend(
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1, 1.18),
        ncol=3,
        )
    _style_time_axis(fluorescence_axis)

    qc_axis = figure.add_subplot(grid[0, 1])
    state = context['recommended_state']
    for state_name, colour, label in (
            ('ambiguous', AMBIGUOUS_COLOUR, 'ambiguous'),
            ('out_of_range', OUT_OF_RANGE_COLOUR, 'out of range'),
            ):
        selected = state == state_name
        qc_axis.scatter(
            timestamps[selected],
            context['displacement_magnitude_px'][selected],
            s=5,
            color=colour,
            alpha=0.8,
            linewidths=0,
            label=label,
            zorder=3,
            )
    qc_axis.plot(
        timestamps,
        context['displacement_magnitude_px'],
        color=NEUTRAL_COLOUR,
        linewidth=0.55,
        alpha=0.8,
        label='rigid translation',
        )
    accepted_count = int(np.count_nonzero(context['analysis_valid']))
    qc_axis.set_title('Registration quality control', loc='left')
    qc_axis.set_xlabel('session time (s)')
    qc_axis.set_ylabel('translation magnitude (px)')
    qc_axis.set_ylim(-2, 76)
    qc_axis.legend(
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1, 1.18),
        ncol=3,
        )
    _style_time_axis(qc_axis)

    aligned_axis = figure.add_subplot(grid[0, 2])
    pulse_duration_s = np.median([
        (stop - start) / context['sampling_frequency_hz']
        for start, stop in events['pulses']
        ])
    pulse_count = len(events['pulses'])
    aligned_axis.axvspan(
        -pulse_duration_s,
        0,
        color='0.88',
        linewidth=0,
        label='stimulation',
        )
    for channel, colour, label in (
            ('signal', SIGNAL_COLOUR, 'dLight'),
            ('control', CONTROL_COLOUR, 'tdTomato'),
            ):
        mean, sem = events['population'][channel]
        aligned_axis.fill_between(
            events['time_s'], mean - sem, mean + sem,
            color=colour, alpha=0.18, linewidth=0)
        aligned_axis.plot(
            events['time_s'], mean, color=colour, linewidth=1.6, label=label)
    aligned_axis.axvline(0, color=NEUTRAL_COLOUR, linewidth=0.8, linestyle=':')
    signal_median = float(np.nanmedian(events['responses']['signal']))
    control_median = float(np.nanmedian(events['responses']['control']))
    aligned_axis.set_title(
        'Stimulation-off aligned median ROI dF/F', loc='left')
    aligned_axis.set_xlabel('time from stimulation off (s)')
    aligned_axis.set_ylabel('dF/F change from local baseline')
    aligned_axis.legend(
        frameon=False,
        title=f'mean ± SEM across {pulse_count} pulses',
        title_fontsize=8,
        loc='upper right',
        bbox_to_anchor=(1, 1.24),
        ncol=3,
        )
    aligned_axis.set_xlim(ALIGNMENT_START_S, ALIGNMENT_STOP_S)
    aligned_axis.set_xticks(np.arange(
        np.ceil(ALIGNMENT_START_S), np.floor(ALIGNMENT_STOP_S) + 1,
        dtype=int,
        ))
    _style_time_axis(aligned_axis)

    heatmap_axis = figure.add_subplot(grid[1, 2])
    response_order = np.argsort(events['responses']['signal'])[::-1]
    heatmap = events['trial_means']['signal'][:, response_order].T
    vmax = float(np.nanpercentile(np.abs(heatmap), 98))
    colour_map = plt.get_cmap('PRGn').copy()
    colour_map.set_bad('0.88')
    image = heatmap_axis.imshow(
        heatmap,
        aspect='auto',
        interpolation='nearest',
        cmap=colour_map,
        vmin=-vmax,
        vmax=vmax,
        extent=(
            events['time_s'][0],
            events['time_s'][-1],
            len(response_order) + 0.5,
            0.5,
            ),
        )
    heatmap_axis.axvline(0, color='0.25', linewidth=0.7, linestyle=':')
    heatmap_axis.set_title('Trial-mean dLight by ROI', loc='left')
    heatmap_axis.set_xlabel('time from stimulation off (s)')
    heatmap_axis.set_ylabel(f'proposed ROIs (n={len(response_order)})')
    heatmap_axis.set_xlim(ALIGNMENT_START_S, ALIGNMENT_STOP_S)
    heatmap_axis.set_xticks(np.arange(
        np.ceil(ALIGNMENT_START_S), np.floor(ALIGNMENT_STOP_S) + 1,
        dtype=int,
        ))
    colour_bar = figure.colorbar(image, ax=heatmap_axis, pad=0.02, fraction=0.046)
    colour_bar.set_label('dF/F change', fontsize=9)
    colour_bar.ax.tick_params(labelsize=8)

    figure.suptitle(
        'Example session',
        x=0.055,
        y=0.97,
        ha='left',
        fontsize=14,
        fontweight='normal',
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=220, facecolor='white')
    plt.close(figure)
    return {
        'output_path': out_path,
        'frame_count': len(timestamps),
        'accepted_frame_count': accepted_count,
        'roi_count': len(rois['roi_dict']),
        'pulse_count': len(events['pulses']),
        'signal_response_median': signal_median,
        'control_response_median': control_median,
        'positive_signal_roi_count': int(np.count_nonzero(
            events['responses']['signal'] > 0)),
        }


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nwb_path', type=Path)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    result = plot_full_session_demo(args.nwb_path, args.out)
    print(f'saved {result["output_path"]}')
    print(
        f'{result["pulse_count"]} pulses, '
        f'{result["positive_signal_roi_count"]}/{result["roi_count"]} positive dLight ROIs')
    print(
        f'median post-offset change: dLight {result["signal_response_median"]:+.3f}, '
        f'tdTomato {result["control_response_median"]:+.3f}')


if __name__ == '__main__':
    main()

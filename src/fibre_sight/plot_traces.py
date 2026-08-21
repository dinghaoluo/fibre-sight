'''
Created on 20 August 2026

plot selected ROI fluorescence and dF/F traces for QA

@author: Dinghao Luo
'''

#%% imports
import argparse
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

from ._formatting import mpl_formatting
from ._repo import FIGURE_ROOT
from .dff import load_dff_run
from .fluorescence import load_fluorescence_run


#%% trace context
DEFAULT_ROI_COUNT = 4


def _trace_context(nwb_path, run_name):
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        nwbfile = io.read()
        dff_container = nwbfile.processing['dff'][run_name]
        timestamps = np.asarray(
            dff_container.roi_response_series['signal_roi_dff'].timestamps)
        analysis_valid = np.asarray(
            nwbfile.processing['quality_control']['registration_qc'][
                'analysis_valid'],
            dtype=bool,
            )
    return timestamps, analysis_valid


def _invalid_intervals(timestamps, analysis_valid):
    timestamps = np.asarray(timestamps, dtype=float)
    analysis_valid = np.asarray(analysis_valid, dtype=bool)
    invalid = ~analysis_valid
    if len(timestamps) == 0 or not invalid.any():
        return []
    changes = np.diff(np.r_[False, invalid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    frame_differences = np.diff(timestamps)
    frame_duration = (
        float(np.median(frame_differences))
        if len(frame_differences) else 1.0
        )
    return [
        (timestamps[start], timestamps[stop - 1] + frame_duration)
        for start, stop in zip(starts, stops)
        ]


def _selected_roi_indices(roi_ids, requested_roi_ids):
    roi_ids = np.asarray(roi_ids, dtype=np.int64)
    if requested_roi_ids is None:
        requested_roi_ids = roi_ids[:DEFAULT_ROI_COUNT]
    if not len(requested_roi_ids):
        raise ValueError('the dF/F run contains no ROIs to plot')
    index_by_roi_id = {int(roi_id): index for index, roi_id in enumerate(roi_ids)}
    missing = [roi_id for roi_id in requested_roi_ids if roi_id not in index_by_roi_id]
    if missing:
        raise ValueError(f'ROI IDs do not exist in the dF/F run: {missing}')
    return [index_by_roi_id[roi_id] for roi_id in requested_roi_ids]


#%% plotting
def plot_dff_traces(nwb_path, run_name, out_path, roi_ids=None):
    dff = load_dff_run(nwb_path, run_name)
    fluorescence = load_fluorescence_run(
        nwb_path,
        dff['fluorescence_run'],
        )
    timestamps, analysis_valid = _trace_context(nwb_path, run_name)
    selected_indices = _selected_roi_indices(dff['roi_ids'], roi_ids)
    selected_roi_ids = [int(dff['roi_ids'][index]) for index in selected_indices]
    provenance = dff['provenance']
    statistic = provenance['statistic']
    coefficient = float(provenance['surround_coefficient'])
    invalid_intervals = _invalid_intervals(timestamps, analysis_valid)
    fluorescence_traces = fluorescence['traces']
    dff_traces = dff['traces']

    import matplotlib.pyplot as plt

    mpl_formatting()
    figure, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(13, 2.3 * len(selected_indices)),
        sharex='col',
        )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.07,
        top=0.80,
        hspace=0.35,
        wspace=0.28,
        )
    axes = np.atleast_2d(axes)
    signal_colour = '#1A1A1C'
    surround_colour = '#D55E00'
    corrected_colour = '#009E73'
    control_colour = '#0072B2'
    signal_control_colour = '#882255'

    for row_index, roi_index in enumerate(selected_indices):
        raw_axis, dff_axis, qc_axis = axes[row_index]
        raw_roi = fluorescence_traces[f'signal_roi_{statistic}'][:, roi_index].copy()
        raw_surround = fluorescence_traces[
            f'signal_surround_{statistic}'][:, roi_index].copy()
        raw_roi[~analysis_valid] = np.nan
        raw_surround[~analysis_valid] = np.nan
        raw_corrected = raw_roi - coefficient * raw_surround
        raw_axis.plot(timestamps, raw_roi, color=signal_colour, label='signal ROI')
        raw_axis.plot(
            timestamps,
            raw_surround,
            color=surround_colour,
            label='signal surround',
            )
        raw_axis.plot(
            timestamps,
            raw_corrected,
            color=corrected_colour,
            linestyle='--',
            label='surround-corrected signal',
            )
        raw_axis.set_ylabel(f'ROI {selected_roi_ids[row_index]}\ncounts')

        dff_axis.plot(
            timestamps,
            dff_traces['signal_surround_corrected_dff'][:, roi_index],
            color=corrected_colour,
            label='signal corrected dF/F',
            )
        dff_axis.plot(
            timestamps,
            dff_traces['control_surround_corrected_dff'][:, roi_index],
            color=control_colour,
            label='control corrected dF/F',
            )
        if 'signal_control_corrected_dff' in dff_traces:
            dff_axis.plot(
                timestamps,
                dff_traces['signal_control_corrected_dff'][:, roi_index],
                color=signal_control_colour,
                label='signal minus control',
                )
        dff_axis.axhline(0, color='0.45', linewidth=0.8)
        dff_axis.set_ylabel('dF/F')

        qc_axis.plot(
            timestamps,
            100 * fluorescence_traces[
                'signal_roi_valid_fraction'][:, roi_index],
            color=signal_colour,
            label='ROI pixel coverage',
            )
        qc_axis.plot(
            timestamps,
            100 * fluorescence_traces[
                'signal_surround_valid_fraction'][:, roi_index],
            color=surround_colour,
            label='surround pixel coverage',
            )
        qc_axis.step(
            timestamps,
            100 * analysis_valid.astype(float),
            where='post',
            color='0.45',
            linestyle=':',
            label='analysis accepted',
            )
        qc_axis.set_ylim(-5, 105)
        qc_axis.set_ylabel('valid pixels (%)')

        for axis in (raw_axis, dff_axis, qc_axis):
            for start, stop in invalid_intervals:
                axis.axvspan(start, stop, color='0.85', alpha=0.55, linewidth=0)
            axis.grid(axis='y', color='0.90', linewidth=0.6)
            axis.spines[['top', 'right']].set_visible(False)

    axes[0, 0].set_title('raw signal and surround, N/A frames masked')
    axes[0, 1].set_title('surround-corrected dF/F, N/A frames masked')
    axes[0, 2].set_title('pixel coverage and analysis QC')
    for axis in axes[-1]:
        axis.set_xlabel('time (s)')
    for column_index, axis in enumerate(axes[0]):
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=((column_index + 0.5) / 3, 0.91),
            frameon=False,
            ncol=1,
            fontsize=8,
            )
    figure.suptitle(
        f'dF/F trace QA: {run_name} | surround coefficient {coefficient:g} | '
        f'{statistic} | {np.count_nonzero(~analysis_valid)} N/A frames',
        fontsize=12,
        y=0.98,
        )
    figure.text(
        0.5,
        0.015,
        'pixel coverage = fraction inside the registered image area; '
        'N/A = preprocessing QC rejected the frame',
        ha='center',
        fontsize=8,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return {
        'output_path': out_path,
        'run_name': run_name,
        'roi_ids': selected_roi_ids,
        'frame_count': len(timestamps),
        'rejected_frame_count': int(np.count_nonzero(~analysis_valid)),
        'invalid_interval_count': len(invalid_intervals),
        }


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nwb_path', type=Path)
    parser.add_argument('run_name')
    parser.add_argument(
        '--out',
        type=Path,
        default=FIGURE_ROOT / 'dff_trace_qa.png',
        )
    parser.add_argument('--roi-id', type=int, action='append', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = plot_dff_traces(
        args.nwb_path,
        args.run_name,
        args.out,
        roi_ids=args.roi_id,
        )
    print(f'saved {result["output_path"]}')
    print(f'ROIs: {result["roi_ids"]}')
    print(f'rejected frames: {result["rejected_frame_count"]}')


if __name__ == '__main__':
    main()

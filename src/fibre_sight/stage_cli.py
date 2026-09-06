'''
Created on 5 September 2026

run the post-registration recording stages from an NWB file

@author: Dinghao Luo
'''

#%% imports
import argparse

from .api import calculate_dff, extract_fluorescence, segment_recording
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


#%% proposal segmentation
def _segment_parser():
    parser = argparse.ArgumentParser(
        description='add a named proposal run to an NWB file',
        )
    parser.add_argument('nwb_path')
    parser.add_argument('--run-name', default='proposal_auto')
    parser.add_argument('--checkpoint')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--min-size', type=int)
    parser.add_argument(
        '--tta',
        action=argparse.BooleanOptionalAction,
        default=True,
        )
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    return parser


def segment_main():
    args = _segment_parser().parse_args()
    result = segment_recording(
        args.nwb_path,
        args.run_name,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        min_size=args.min_size,
        tta=args.tta,
        device=args.device,
        )
    print(f'segmentation completed: {result["roi_count"]} ROI proposals', flush=True)


#%% fluorescence extraction
def _fluorescence_parser():
    parser = argparse.ArgumentParser(
        description='extract ROI and surround fluorescence from an NWB file',
        )
    parser.add_argument('nwb_path')
    parser.add_argument('--run-name', default='fluorescence_auto')
    parser.add_argument('--roi-run', default='proposal_auto')
    parser.add_argument('--surround-method', choices=('adaptive', 'fixed'), default=SURROUND_METHOD)
    parser.add_argument('--surround-inner-px', type=int, default=SURROUND_INNER_PX)
    parser.add_argument('--surround-outer-px', type=int, default=SURROUND_OUTER_PX)
    parser.add_argument('--surround-min-pixels', type=int, default=SURROUND_MIN_PIXELS)
    return parser


def fluorescence_main():
    args = _fluorescence_parser().parse_args()
    result = extract_fluorescence(
        args.nwb_path,
        args.run_name,
        args.roi_run,
        surround_method=args.surround_method,
        surround_inner_px=args.surround_inner_px,
        surround_outer_px=args.surround_outer_px,
        surround_min_pixels=args.surround_min_pixels,
        )
    print(f'fluorescence completed: {result["roi_count"]} ROIs', flush=True)


#%% dF/F calculation
def _dff_parser():
    parser = argparse.ArgumentParser(
        description='calculate a named dF/F run in an NWB file',
        )
    parser.add_argument('nwb_path')
    parser.add_argument('--run-name', default='dff_auto')
    parser.add_argument('--fluorescence-run', default='fluorescence_auto')
    parser.add_argument('--statistic', choices=('mean', 'median'), default=STATISTIC)
    parser.add_argument('--baseline-percentile', type=float, default=BASELINE_PERCENTILE)
    parser.add_argument('--baseline-window-s', type=float, default=BASELINE_WINDOW_S)
    parser.add_argument('--surround-coefficient', type=float, default=SURROUND_COEFFICIENT)
    parser.add_argument('--control-correction', choices=('none', 'subtract_dff'), default=CONTROL_CORRECTION)
    return parser


def dff_main():
    args = _dff_parser().parse_args()
    result = calculate_dff(
        args.nwb_path,
        args.run_name,
        args.fluorescence_run,
        statistic=args.statistic,
        baseline_percentile=args.baseline_percentile,
        baseline_window_s=args.baseline_window_s,
        surround_coefficient=args.surround_coefficient,
        control_correction=args.control_correction,
        )
    print(f'dF/F completed: {result["roi_count"]} ROI traces', flush=True)

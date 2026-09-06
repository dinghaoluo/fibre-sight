'''
Created on 5 September 2026

run the registration stage from TIFF folders

@author: Dinghao Luo
'''

#%% imports
import argparse
from pathlib import Path

from .gui_worker import natural_tiff_paths
from .preprocessing import (
    SEGMENTATION_REFERENCE_PERCENTILES,
    preprocess_recording,
    )


#%% command line
def parse_args():
    parser = argparse.ArgumentParser(
        description='register a FibreSight recording and write an NWB file',
        )
    parser.add_argument('tiff_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-tiff-dir', type=Path)
    parser.add_argument('--layout', choices=('interleaved', 'separate'), default='interleaved')
    parser.add_argument('--signal-channel', type=int, default=1)
    parser.add_argument('--control-channel', type=int, default=2)
    parser.add_argument('--sampling-frequency', type=float, default=30)
    parser.add_argument('--signal-label', default='signal')
    parser.add_argument('--control-label', default='control')
    parser.add_argument('--pixel-size', type=float)
    parser.add_argument('--registration-model', choices=('rigid', 'piecewise', 'auto'), default='auto')
    parser.add_argument('--registration-channel', choices=('signal', 'control'), default='control')
    parser.add_argument('--reference-channel', choices=('signal', 'control'), default='control')
    parser.add_argument('--reference-low-percentile', type=float, default=SEGMENTATION_REFERENCE_PERCENTILES[0])
    parser.add_argument('--reference-high-percentile', type=float, default=SEGMENTATION_REFERENCE_PERCENTILES[1])
    return parser.parse_args()


def main():
    args = parse_args()
    signal_dir = args.tiff_dir.expanduser().resolve()
    if not signal_dir.is_dir():
        raise SystemExit(f'TIFF folder does not exist: {signal_dir}')
    signal_paths = natural_tiff_paths(signal_dir)
    if not signal_paths:
        raise SystemExit(f'no TIFF files found in {signal_dir}')

    multiplexed = args.layout == 'interleaved'
    control_paths = None
    if not multiplexed:
        if args.control_tiff_dir is None:
            raise SystemExit('--control-tiff-dir is required for separate TIFF folders')
        control_dir = args.control_tiff_dir.expanduser().resolve()
        if not control_dir.is_dir():
            raise SystemExit(f'TIFF folder does not exist: {control_dir}')
        control_paths = natural_tiff_paths(control_dir)
        if len(control_paths) != len(signal_paths):
            raise SystemExit('signal and control TIFF folders contain different file counts')

    preprocess_recording(
        signal_paths,
        args.output.expanduser().resolve(),
        signal_channel=args.signal_channel,
        control_channel=args.control_channel,
        multiplexed=multiplexed,
        sampling_frequency_hz=args.sampling_frequency,
        signal_label=args.signal_label,
        control_label=args.control_label,
        control_tiff_paths=control_paths,
        registration_model=args.registration_model,
        registration_channel=args.registration_channel,
        pixel_size_um=args.pixel_size,
        segmentation_reference_channel=args.reference_channel,
        segmentation_reference_percentiles=(
            args.reference_low_percentile,
            args.reference_high_percentile,
            ),
        )
    print(f'registration completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()

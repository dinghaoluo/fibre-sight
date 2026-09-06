'''
Created on 20 August 2026

list immutable analysis runs in an NWB file

@author: Dinghao Luo
'''

#%% imports
import argparse
from pathlib import Path

from pynwb import NWBHDF5IO

from .dff import _dff_runs
from .fluorescence import _fluorescence_runs
from .nwb_segmentation import _segmentation_runs


#%% listing
RUN_KINDS = ('all', 'roi', 'fluorescence', 'dff')
TABLE_COLUMNS = (
    'kind',
    'run_name',
    'run_type',
    'source_run',
    'roi_run',
    'fluorescence_run',
    'statistic',
    'created_at',
    )


def list_analysis_runs(nwb_path, kind='all', run_type=None):
    if kind not in RUN_KINDS:
        raise ValueError(f'kind must be one of {RUN_KINDS}')

    runs = []
    with NWBHDF5IO(Path(nwb_path), 'r') as io:
        nwbfile = io.read()
        if kind in ('all', 'roi'):
            roi_runs = _segmentation_runs(nwbfile).values()
            if run_type is not None:
                roi_runs = [
                    run for run in roi_runs if run['run_type'] == run_type]
            runs.extend({'kind': 'roi', **run} for run in roi_runs)
        if kind in ('all', 'fluorescence'):
            runs.extend(
                {'kind': 'fluorescence', **run}
                for run in _fluorescence_runs(nwbfile).values()
                )
        if kind in ('all', 'dff'):
            runs.extend(
                {'kind': 'dff', **run}
                for run in _dff_runs(nwbfile).values()
                )
    return runs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nwb_path', type=Path)
    parser.add_argument('--kind', choices=RUN_KINDS, default='all')
    parser.add_argument('--run-type', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runs = list_analysis_runs(
        args.nwb_path,
        kind=args.kind,
        run_type=args.run_type,
        )
    print('\t'.join(TABLE_COLUMNS))
    for run in runs:
        print('\t'.join(str(run.get(column, '')) for column in TABLE_COLUMNS))


if __name__ == '__main__':
    main()

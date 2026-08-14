'''
Created on 7 April 2026

Modified on 2 June 2026
Modified on 23 June 2026
Modified on 24 July 2026 to use the standalone workspace
Modified on 14 August 2026

scan labelled sessions and write the training table

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import argparse

from ._repo import SOURCE_ROOT, WORKSPACE_ROOT
from .manifest import (
    assign_session_splits,
    scan_source_root,
    write_manifest,
    )


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        '--out',
        type=Path,
        default=WORKSPACE_ROOT / 'manifests' / 'ch2_manifest.csv',
        )
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-splits', action='store_true')
    return parser.parse_args()


def print_summary(sessions):
    included = [session for session in sessions if session['included']]
    animals = sorted({session['animal'] for session in included})
    total_rois = sum(session['roi_count'] for session in included)

    print(f'total sessions: {len(sessions)}')
    print(f'included sessions: {len(included)}')
    print(f'animals: {animals}')
    print(f'total ROIs: {total_rois}')

    split_counts = {}
    for session in included:
        split = session['split']
        split_counts[split] = split_counts.get(split, 0) + 1
    if split_counts:
        print(f'splits: {split_counts}')


def main():
    args = parse_args()
    sessions = scan_source_root(args.source_root)

    if not args.no_splits:
        sessions = assign_session_splits(
            sessions,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            )

    write_manifest(sessions, args.out)
    print_summary(sessions)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()

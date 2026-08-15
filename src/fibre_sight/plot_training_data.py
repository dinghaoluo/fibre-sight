'''
Created on 11 April 2026

Modified on 2 June 2026
Modified on 23 June 2026
Modified on 24 July 2026 to use local figure and manifest paths
Modified on 14 August 2026

inspect channel-2 training images with their curated labels

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import argparse

import numpy as np

from ._formatting import mpl_formatting
from ._repo import FIGURE_ROOT
from .image_ops import normalise
from .manifest import read_manifest
from .roi_io import load_roi_dict, roi_dict_to_label


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', default=None)
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--out',
        type=Path,
        default=FIGURE_ROOT / 'diagnostics' / 'training_label_overlays.png',
        )
    return parser.parse_args()


#%% plotting
def make_overlay(image, labelled, alpha=0.55):
    base = normalise(image)
    out = np.dstack([base, base, base])

    if labelled.max() == 0:
        return out

    # stable colours make repeat previews comparable while the sampled sessions change
    rng = np.random.default_rng(42)
    colours = rng.uniform(0.1, 1.0, size=(int(labelled.max()) + 1, 3))
    overlay = colours[labelled]
    mask = labelled > 0
    out[mask] = (1 - alpha) * out[mask] + alpha * overlay[mask]

    boundary = mask & (
        (np.roll(labelled, 1, axis=0) != labelled) |
        (np.roll(labelled, -1, axis=0) != labelled) |
        (np.roll(labelled, 1, axis=1) != labelled) |
        (np.roll(labelled, -1, axis=1) != labelled)
        )
    out[boundary] = [1, 0.95, 0.05]
    return out


def choose_sessions(sessions, n_sessions, seed=42):
    if len(sessions) <= n_sessions:
        return sessions

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sessions), size=n_sessions, replace=False)
    return [sessions[int(i)] for i in idx]


def plot_sessions(sessions, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mpl_formatting()

    n_rows = len(sessions)
    n_cols = min(3, n_rows)
    n_fig_rows = int(np.ceil(n_rows / n_cols))

    fig, axes = plt.subplots(
        n_fig_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.5 * n_fig_rows),
        constrained_layout=True,
        )
    axes = np.atleast_1d(axes).ravel()

    for ax, session in zip(axes, sessions):
        image = np.load(session['image_path'])
        roi_dict = load_roi_dict(session['roi_path'])
        labelled, _ = roi_dict_to_label(roi_dict, image.shape)
        name = session['session']
        roi_count = session['roi_count']
        positive_fraction = session['positive_fraction']

        ax.imshow(make_overlay(image, labelled), interpolation='nearest')
        ax.set_title(
            f'{name}\n'
            f'{roi_count} ROIs, {100 * positive_fraction:.2f}% labelled',
            fontsize=9,
            )
        ax.set_axis_off()

    for ax in axes[len(sessions):]:
        ax.set_axis_off()

    fig.suptitle('channel-2 references with curated axon ROIs', fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    sessions = read_manifest(args.manifest, included_only=True, split=args.split)
    sessions = choose_sessions(sessions, args.n, seed=args.seed)
    plot_sessions(sessions, args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()

'''
Created on 23 April 2026
Modified on 2 June 2026
Modified on 23 June 2026
Modified on 23 July 2026 to keep held-out scoring in this script
Modified on 24 July 2026 to use the bundled model and local output paths
score a trained model on labelled sessions

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import argparse
import csv

import numpy as np

from ._device import resolve_device
from ._formatting import print_files_saved
from .manifest import read_manifest
from .postprocess import probability_to_labels
from .predict_rois import load_trained_model, predict_probability
from .roi_io import load_roi_dict, roi_dict_to_label


#%% scoring
# Pixel overlap and ROI counts answer different questions, so I kept both.
def mask_scores(pred_mask, target_mask, eps=1e-8):
    pred_mask = np.asarray(pred_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)

    tp = np.sum(pred_mask & target_mask)
    fp = np.sum(pred_mask & ~target_mask)
    fn = np.sum(~pred_mask & target_mask)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f2 = 5 * tp / (5 * tp + 4 * fn + fp + eps)

    return {
        'dice': float(dice),
        'f2': float(f2),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'true_positive_pixels': int(tp),
        'false_positive_pixels': int(fp),
        'false_negative_pixels': int(fn),
        }


def component_counts(pred_labelled, target_labelled):
    pred_ids = set(np.unique(pred_labelled)) - {0}
    target_ids = set(np.unique(target_labelled)) - {0}

    return {
        'predicted_components': len(pred_ids),
        'target_components': len(target_ids),
        }


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--min-size', type=int, default=None)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


#%% evaluation
def evaluate_rows(rows, checkpoint_path, threshold=None, min_size=None, device='auto', tta=False):
    device = resolve_device(device)

    model, checkpoint = load_trained_model(checkpoint_path, device)
    data_cfg = checkpoint.get('data_config', {})
    post_cfg = dict(checkpoint.get('postprocess_config', {}))

    if threshold is not None:
        post_cfg['threshold'] = threshold
    if min_size is not None:
        post_cfg['min_size'] = min_size

    results = []
    for row in rows:
        image = np.load(row['image_path'])
        roi_dict = load_roi_dict(row['roi_path'])
        target_labelled, _, _ = roi_dict_to_label(roi_dict, image.shape)
        target_mask = target_labelled > 0

        probability = predict_probability(
            image,
            model,
            device,
            normalise_percentiles=data_cfg.get('normalise_percentiles', [1, 99.7]),
            tta=tta,
            )
        # held-out scoring uses the same postprocessing as saved ROI output
        pred_labelled = probability_to_labels(
            probability,
            threshold=post_cfg.get('threshold', 0.5),
            min_size=post_cfg.get('min_size', 30),
            max_size=post_cfg.get('max_size', None),
            )
        scores = mask_scores(pred_labelled > 0, target_mask > 0)
        # Pixel overlap misses split and merged ROIs, so keep the component counts beside it.
        scores.update(component_counts(pred_labelled, target_labelled))
        scores.update({
            'session': row['session'],
            'animal': row['animal'],
            'split': row['split'],
            })
        results.append(scores)

    return results


def write_results(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results):
    if len(results) == 0:
        print('no rows evaluated')
        return

    for key in ['dice', 'f2', 'iou', 'precision', 'recall']:
        values = [row[key] for row in results]
        print(f'{key}: {np.mean(values):.4f} +/- {np.std(values):.4f}')


def main():
    args = parse_args()
    rows = read_manifest(args.manifest, included_only=True, split=args.split)
    results = evaluate_rows(
        rows,
        args.checkpoint,
        threshold=args.threshold,
        min_size=args.min_size,
        device=args.device,
        tta=args.tta,
        )

    out_path = args.out
    if out_path is None:
        out_path = Path(args.checkpoint).parent / f'evaluate_{args.split}.csv'

    write_results(results, out_path)
    print_summary(results)
    print_files_saved([
        ('evaluation', out_path),
    ])


if __name__ == '__main__':
    main()

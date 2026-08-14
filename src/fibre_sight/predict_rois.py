'''
Created on 22 April 2026

Modified on 23 June 2026
Modified on 24 July 2026 to use the bundled checkpoint by default
Modified on 14 August 2026

run a trained model on channel-2 references and save ROI dictionaries

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import argparse

import numpy as np
import torch

from ._device import get_device
from ._repo import PACKAGE_ROOT
from .image_ops import normalise
from .model import build_model
from .postprocess import probability_to_roi_dict
from .roi_io import save_roi_dict


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=PACKAGE_ROOT / 'models' / 'fibre_sight_ch2_v1.pt',
        )
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--min-size', type=int, default=None)
    parser.add_argument(
        '--tta',
        action=argparse.BooleanOptionalAction,
        default=True,
        )
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


#%% prediction
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_probability(image, model, device, normalise_percentiles=(1, 99.7), tta=False):
    image = normalise(
        image,
        low=normalise_percentiles[0],
        high=normalise_percentiles[1],
        )
    tensor = torch.from_numpy(image[None, None, ...]).to(device)

    with torch.no_grad():
        probability = _predict_tensor(tensor, model, tta=tta)
    return probability[0, 0].detach().cpu().numpy()


def _predict_tensor(tensor, model, tta=False):
    if not tta:
        return torch.sigmoid(model(tensor)[:, :1])

    # average the four flip views without interpolating the thin ROI shapes
    predictions = []
    for dims in [(), (2,), (3,), (2, 3)]:
        input_tensor = torch.flip(tensor, dims=dims) if dims else tensor
        probability = torch.sigmoid(model(input_tensor)[:, :1])
        predictions.append(torch.flip(probability, dims=dims) if dims else probability)
    return torch.mean(torch.stack(predictions), dim=0)


def predict_roi_dict(
        image_path,
        checkpoint_path,
        out_path=None,
        threshold=None,
        min_size=None,
        device='auto',
        tta=False,
        ):
    device = get_device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    postprocess = checkpoint['postprocess_config']
    if threshold is None:
        threshold = postprocess['threshold']
    if min_size is None:
        min_size = postprocess['min_size']

    image = np.load(image_path)
    probability = predict_probability(
        image,
        model,
        device,
        normalise_percentiles=checkpoint['data_config']['normalise_percentiles'],
        tta=tta,
        )
    roi_dict, labelled = probability_to_roi_dict(
        probability,
        threshold=threshold,
        min_size=min_size,
        max_size=postprocess['max_size'],
        )

    if out_path is None:
        recname = Path(image_path).stem.removesuffix('_ref_mat_ch2')
        out_path = Path(image_path).parent / f'{recname}_ROI_dict.npy'

    save_roi_dict(roi_dict, out_path)
    return roi_dict, labelled, probability


def main():
    args = parse_args()
    roi_dict, _, _ = predict_roi_dict(
        args.image,
        args.checkpoint,
        out_path=args.out,
        threshold=args.threshold,
        min_size=args.min_size,
        device=args.device,
        tta=args.tta,
        )
    print(f'predicted ROIs: {len(roi_dict)}')


if __name__ == '__main__':
    main()

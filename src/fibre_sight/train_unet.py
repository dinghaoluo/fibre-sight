'''
Created on 10 April 2026

Modified on 19 May 2026
Modified on 23 June 2026
Modified on 23 July 2026 to keep the loss experiments beside the training loop
Modified on 24 July 2026 to use packaged recipes and local run directories
Modified on 1 August 2026 to seed comparable runs and expose augmentation choices
Modified on 14 August 2026

train the small U-Net and keep its run record

@author: Dinghao Luo
'''

#%% imports
from contextlib import nullcontext
from pathlib import Path
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._device import get_device
from ._formatting import mpl_formatting
from ._repo import FIGURE_ROOT, PACKAGE_ROOT, WORKSPACE_ROOT
from .config import load_recipe, resolve_path, save_recipe
from .dataset import ROIDataset
from .model import build_model

#%% loss
# the first baseline used BCE; Dice helped with the sparse foreground
def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))

    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def soft_tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))

    tp = torch.sum(probs * targets, dim=dims)
    fp = torch.sum(probs * (1 - targets), dim=dims)
    fn = torch.sum((1 - probs) * targets, dim=dims)

    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - score.mean()


def segmentation_loss(logits, targets, config):
    mode = config['mode']
    channel_weights = config.get('channel_weights', None)
    pos_weight = config.get('pos_weight', None)

    if channel_weights is not None:
        weights = torch.as_tensor(channel_weights, device=logits.device, dtype=logits.dtype)
        weights = weights.view(1, -1, 1, 1)
    else:
        weights = 1.0

    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        pos_weight = pos_weight.view(1, -1, 1, 1)

    bce_raw = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction='none',
        )
    bce = torch.mean(bce_raw * weights)

    if mode == 'bce_dice':
        seg = soft_dice_loss(logits * weights, targets)
        seg_name = 'dice_loss'
    elif mode == 'bce_tversky':
        seg = soft_tversky_loss(
            logits,
            targets,
            alpha=config['tversky_alpha'],
            beta=config['tversky_beta'],
            )
        seg_name = 'tversky_loss'
    else:
        raise ValueError(f'unknown loss mode: {mode}')

    loss = config['bce_weight'] * bce + config['seg_weight'] * seg
    return loss, {
        'loss': float(loss.detach().cpu()),
        'bce': float(bce.detach().cpu()),
        seg_name: float(seg.detach().cpu()),
        }


#%% cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=PACKAGE_ROOT / 'configs' / 'ch2_unet.yaml',
        )
    return parser.parse_args()


#%% setup
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataloaders(config, device):
    data = config['data']
    training = config['train']
    manifest_path = resolve_path(data['manifest'], WORKSPACE_ROOT)
    normalise_percentiles = data['normalise_percentiles']
    augmentation = data['augmentation']

    train_dataset = ROIDataset(
        manifest_path,
        split=data['train_split'],
        patch_size=data['patch_size'],
        patches_per_image=data['patches_per_image'],
        foreground_fraction=data['foreground_fraction'],
        normalise_percentiles=normalise_percentiles,
        augment=True,
        rotation_90=augmentation['rotation_90'],
        noise_sd=augmentation['noise_sd'],
        cache_images=data['cache_images'],
        target_mode=data['target_mode'],
        support_radius=data.get('support_radius', 3),
        seed=training['seed'],
        )
    val_dataset = ROIDataset(
        manifest_path,
        split=data['val_split'],
        patch_size=data['patch_size'],
        patches_per_image=data['val_patches_per_image'],
        foreground_fraction=data['foreground_fraction'],
        normalise_percentiles=normalise_percentiles,
        augment=False,
        cache_images=data['cache_images'],
        target_mode=data['target_mode'],
        support_radius=data.get('support_radius', 3),
        seed=training['seed'] + 1000,
        )

    loader_args = {
        'batch_size': training['batch_size'],
        'num_workers': training['num_workers'],
        'pin_memory': training['pin_memory'] and device.type == 'cuda',
        }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    # a fixed crop set makes validation epochs directly comparable
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    return train_dataset, val_dataset, train_loader, val_loader


def prepare_run_dir(config):
    training = config['train']
    run_dir = resolve_path(training['out_dir'], WORKSPACE_ROOT) / training['run_name']
    run_dir.mkdir(parents=True)
    save_recipe(config, run_dir / 'config.yaml')
    return run_dir


#%% training
def train_one_epoch(model, loader, optimiser, scaler, device, config, epoch):
    model.train()
    loader.dataset.set_epoch(epoch)
    batches = []
    amp = use_amp(config, device)

    for batch in tqdm(loader, desc=f'train {epoch}', leave=False):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        with autocast_context(amp):
            logits = model(images)
            loss, loss_parts = segmentation_loss(
                logits,
                masks,
                config['loss'],
                )

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        batches.append(loss_parts)

    return average_batches(batches)


def validate(model, loader, device, config, epoch):
    model.eval()
    batches = []
    amp = use_amp(config, device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'val {epoch}', leave=False):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            with autocast_context(amp):
                logits = model(images)
                _, loss_parts = segmentation_loss(
                    logits,
                    masks,
                    config['loss'],
                    )

            loss_parts['pixel_dice'] = batch_dice(logits, masks)
            batches.append(loss_parts)

    return average_batches(batches)


def batch_dice(logits, masks, threshold=0.5, eps=1e-8):
    pred = torch.sigmoid(logits[:, :1]) >= threshold
    target = masks[:, :1] >= 0.5
    dims = tuple(range(1, pred.ndim))
    tp = torch.sum(pred & target, dim=dims)
    fp = torch.sum(pred & ~target, dim=dims)
    fn = torch.sum(~pred & target, dim=dims)
    dice = (2 * tp.float() + eps) / (2 * tp.float() + fp.float() + fn.float() + eps)
    return float(dice.mean().detach().cpu())


def use_amp(config, device):
    return config['train']['amp'] and device.type == 'cuda'


def autocast_context(enabled):
    if not enabled:
        return nullcontext()

    return torch.amp.autocast('cuda')


#%% output
def save_checkpoint(path, model, optimiser, epoch, best_score, config):
    checkpoint = {
        'epoch': epoch,
        'best_score': best_score,
        'model_state': model.state_dict(),
        'optimiser_state': optimiser.state_dict(),
        'model_config': config['model'],
        'data_config': config['data'],
        'postprocess_config': config['postprocess'],
        'config': config,
        }
    torch.save(checkpoint, path)


def write_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path):
    import matplotlib.pyplot as plt

    mpl_formatting()

    epochs = [stats['epoch'] for stats in history]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    axes[0].plot(epochs, [stats['train_loss'] for stats in history], label='train')
    axes[0].plot(epochs, [stats['val_loss'] for stats in history], label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [stats['val_pixel_dice'] for stats in history])
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('validation Dice')

    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def average_batches(batches):
    keys = batches[0].keys()
    return {
        key: float(np.mean([batch[key] for batch in batches]))
        for key in keys
        }


#%% main
def main():
    args = parse_args()
    config = load_recipe(args.config)
    training = config['train']
    # 1 August 2026: seed model initialisation and loader shuffling for cleaner comparisons
    set_random_seed(training['seed'])
    device = get_device(training['device'])
    run_dir = prepare_run_dir(config)
    figure_run_dir = FIGURE_ROOT / 'runs' / run_dir.name
    figure_run_dir.mkdir(parents=True, exist_ok=True)

    print(f'training on {device}')
    print(f'run directory: {run_dir}')

    train_dataset, val_dataset, train_loader, val_loader = make_dataloaders(config, device)
    print(f'train sessions: {len(train_dataset.sessions)}')
    print(f'validation sessions: {len(val_dataset.sessions)}')

    model = build_model(config['model']).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=training['learning_rate'],
        weight_decay=training['weight_decay'],
        )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp(config, device))

    history = []
    best_score = -np.inf
    epochs = training['epochs']

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimiser, scaler, device, config, epoch)
        val_stats = validate(model, val_loader, device, config, epoch)

        epoch_stats = {'epoch': epoch}
        epoch_stats.update({f'train_{key}': value for key, value in train_stats.items()})
        epoch_stats.update({f'val_{key}': value for key, value in val_stats.items()})
        history.append(epoch_stats)

        val_score = epoch_stats['val_pixel_dice']
        train_loss = epoch_stats['train_loss']
        val_loss = epoch_stats['val_loss']
        print(
            f'epoch {epoch:03d} | '
            f'train loss {train_loss:.4f} | '
            f'val loss {val_loss:.4f} | '
            f'val Dice {val_score:.4f}'
            )

        # keep the checkpoint with the best validation Dice, the mask overlap I inspect
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(run_dir / 'best.pt', model, optimiser, epoch, best_score, config)
        # best.pt is used for prediction; latest.pt records where each long run ended
        save_checkpoint(run_dir / 'latest.pt', model, optimiser, epoch, best_score, config)

        write_history(history, run_dir / 'history.csv')
        plot_history(history, figure_run_dir / 'history.png')


if __name__ == '__main__':
    main()

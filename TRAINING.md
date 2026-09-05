# How to Train Your Model™

The bundled checkpoint covers only our lab's data. It has worked for multiple sensors and animals, but a different indicator, microscope, or labelling convention may need its own model. The entire training workflow is included.

## label sessions

Select a session's `channel-2 image` in the workbench, then open `segment` and run `SEGMENT` to generate MSER proposals. Delete, merge, or fix those proposals, then click `export ROIs`. The current workbench cannot draw a missed fibre from scratch; add any missing ROIs in an external editor and import the resulting dictionary before export.

Arrange the results so each session directory holds both files:

```
workspace/labelled_sessions/<session>/processed_data/ref_mat_ch2.npy
workspace/labelled_sessions/<session>/processed_data/<name>_ROI_dict.npy
```

The reference array and the ROI dictionary must describe the same image.

A few practical notes from our own labelling rounds: dim, out-of-plane processes are generally better left unlabelled, since their fluorescence spreads into the background and cannot be assigned cleanly to one axonal ROI for later analysis. The target describes the in-plane processes which a curator would retain; fibre-like traces outside that convention remain background. Labelling 30-50 sessions across multiple animals is a reasonable starting point; the released checkpoint used 53 sessions and 1,296 curated ROIs.

## build a manifest

```
build-manifest
```

This scans the labelled sessions, writes `workspace/manifests/ch2_manifest.csv`, and prints the number of included sessions, the animals found, the total ROI count, and the split sizes. Splits are assigned at session level with a 15% validation and 15% test fraction by default:

```
build-manifest --val-fraction 0.2 --test-fraction 0.2 --seed 42
```

Use `--no-splits` to write the manifest without splits and assign an animal-held-out split manually in the CSV. `--source-root` and `--out` move the input and output locations.

## write a training recipe

Copy a shipped config and edit it without modifying the packaged file:

```
cp src/fibre_sight/configs/ch2_unet.yaml workspace/my_recipe.yaml
```

Paths inside the config resolve against the workspace root, so `data.manifest` of `manifests/ch2_manifest.csv` and `train.out_dir` of `output/runs` need no change at the default locations.

### recipe anatomy

A recipe is a single YAML with four sections:

**`data`** controls what the training loader sees. For a new dataset, adjust `patch_size` and `patches_per_image` to the field size, `foreground_fraction` to the mask density, and `normalise_percentiles` to the useful intensity range. Every training patch can receive independent horizontal and vertical flips, a lossless quarter-turn rotation, intensity scaling within ±15%, an intensity offset within ±0.0375, and Gaussian noise. The `augmentation` sub-section exposes the rotation and noise choices so that those ablations do not require code edits:

```yaml
data:
  augmentation:
    rotation_90: true    # 90-degree lossless rotations
    noise_sd: 0.02       # Gaussian noise after normalisation
```

Setting `rotation_90: false` disables the quarter-turn rotations. Horizontal and vertical flips are separate fixed loader operations, each applied with probability `0.5`; the intensity jitter is also fixed loader behaviour. Setting `noise_sd: 0` disables additive noise entirely. Validation patches are not augmented.

**`model`** sets the architecture. `base_channels` and `depth` control the network width and resolution levels; `attention_gates: true` activates the attention-gated skip connections (see [alternative architectures](#alternative-architectures) below).

**`loss`** configures the segmentation loss. The default `bce_dice` mode combines binary cross-entropy with soft Dice; `bce_tversky` replaces soft Dice with Tversky loss for recall-heavy training. `channel_weights` and `pos_weight` allow per-channel weighting when multi-channel targets are used.

**`train`** sets the run name, output directory, learning rate, batch size, epoch count, device, and random seed. The seed controls model initialisation, loader shuffling, crop sampling, and augmentation draws, which supports direct comparisons between recipes on the same software and hardware stack.

### device selection

Set `train.device` to `mps`, `cuda`, or `cpu` when automatic selection is unsuitable. CUDA uses automatic mixed precision when `train.amp` is enabled; MPS training stays in float32, with pinned-memory loading disabled because it is CUDA-specific.

## train

```
train --config workspace/my_recipe.yaml
```

Each epoch prints training loss, validation loss, and validation Dice, and writes into `workspace/output/runs/<run_name>/`:

- `best.pt`, the weights with the highest mean validation patch Dice, measured at a probability threshold of `0.5` before any component filtering. This is the checkpoint to use for prediction.
- `latest.pt`, the final checkpoint.
- `history.csv`, rewritten every epoch, alongside a `history.png` under the figures directory.
- `config.yaml`, a copy of the recipe used for the run.

Selection on validation Dice is fixed in the training code, so the saved epoch is often not the epoch where validation loss bottoms out; for the released model those were epoch 37 and epoch 59.

## evaluate on the held-out split

```
evaluate \
  --manifest workspace/manifests/ch2_manifest.csv \
  --checkpoint workspace/output/runs/<run_name>/best.pt \
  --split test \
  --tta \
  --out workspace/output/test_metrics.csv
```

This reports Dice, intersection over union, precision, recall, and F2 per session. `--threshold` and `--min-size` override the postprocessing values for sweeping an operating point; unset, they fall back to the recipe defaults.

Choose the operating point after the weights are fixed, and inspect overlays alongside the numbers. Pixel scores describe foreground overlap and hide split and merge errors, so a setting that looks better on Dice can produce visibly worse ROIs. The [methods record](METHODS.md#choosing-the-released-operating-point) shows the sweep behind the bundled `0.25 / 45` default and why it sits between the recall-heavy and Dice-heavy optima.

## use the new checkpoint

```
predict \
  --image path/to/ref_mat_ch2.npy \
  --checkpoint workspace/output/runs/<run_name>/best.pt \
  --threshold 0.25 \
  --min-size 45 \
  --out workspace/output/predicted_ROI_dict.npy
```

Add `--no-tta` to disable four-view flip averaging, which is on by default. In the GUI, use `browse` beside the global `trained model` field to point at the new `best.pt` instead of the bundled checkpoint.

A new checkpoint that predicts well enough to seed curation replaces MSER as the proposal route for the next labelling round.

## alternative architectures

### attention-gated U-Net

The `ch2_unet_attention.yaml` recipe adds attention gates to the skip connections of the standard `SmallUNet`. The gate projects skip features and the upsampled decoder features into a shared low-dimensional attention space, applies a sigmoid, and multiplies the result back onto the skip path before concatenation. The decoder can then suppress irrelevant skip features (background texture, out-of-focus processes) without the main decoder pathway being modified.

To use it:

```yaml
model:
  in_channels: 1
  out_channels: 1
  base_channels: 24
  depth: 4
  attention_gates: true
```

In our tests, the attention-gated variant performed comparably to the standard U-Net on the same data and training recipe (same seed, same augmentation, same loss). The attention gates add a small number of parameters but do not change the decoder's spatial resolution or output format; a checkpoint trained with `attention_gates: true` loads into the same `SmallUNet` class with the flag set, and prediction works identically.

The recipe is included for cases where the training data contains more heterogeneous backgrounds (different microscopes, varying signal-to-noise) and the skip connections might benefit from learned gating. For our dLight channel-2 images the standard model was sufficient, and the released checkpoint uses the ungated architecture.

### wider 512-pixel model

`ch2_unet_512.yaml` doubles the crop size to 512 pixels and widens the base channels to 32. Larger crops give each patch more spatial context at the cost of fewer independent patches per epoch; wider channels increase capacity at the cost of memory. This trial was a sanity check that the 256-pixel baseline was not leaving performance on the table; the 256-pixel model remained the released checkpoint.

### recall-heavy Tversky trial

`ch2_unet_aux_recall.yaml` added a dilated support target and replaced soft Dice with Tversky loss (`alpha=0.15, beta=0.85`), penalising false negatives far more than false positives. It also used a two-channel output (foreground + dilated support), wider base channels (32), and a higher foreground crop fraction (0.85). This was the recall-heavy extreme; the released postprocessing already favours recall through a low confidence threshold, so the additional recall from a modified loss did not justify the added complexity.

## shipped recipes

| recipe | architecture | crop | base width | loss | notes |
| --- | --- | ---: | ---: | --- | --- |
| `ch2_unet.yaml` | SmallUNet | 256 | 24 | BCE + Dice | released baseline |
| `ch2_unet_512.yaml` | SmallUNet | 512 | 32 | BCE + Dice | wider context trial |
| `ch2_unet_aux_recall.yaml` | SmallUNet (2-ch out) | 256 | 32 | BCE + Tversky | recall-heavy trial |
| `ch2_unet_attention.yaml` | SmallUNet + attention gates | 256 | 24 | BCE + Dice | attention ablation |

All four recipes use the same manifest path and workspace layout. The comparison stopped with these working trials whilst the labelling and curation loop was still changing; no exhaustive architecture search was run.

## reproducibility

Training runs are seeded by `train.seed` in the recipe. The seed controls `numpy` and `torch` random state before model initialisation, so two runs with the same recipe and seed begin with the same initial weights and use the same indexed crop and augmentation sequence. This supports single-variable comparisons (attention gates on vs. off, different loss modes): change one thing in the YAML and keep the seed fixed. Accelerator kernels and differences between software or hardware stacks can still introduce numerical variation.

The saved recipe records `rotation_90` and `noise_sd`. The loader fixes the two flip probabilities and the intensity jitter; YAML does not expose them, so reconstructing the complete augmentation policy also requires the corresponding FibreSight source version.

# FibreSight

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

With tens of sessions and hundreds of neurite ROIs, manual curation can take days of effort, and the resulting segmentation also varies between experimenters. FibreSight reads two-dimensional channel-2 reference arrays (`ref_mat_ch2.npy`) and predicts candidate ROIs; the workbench combines a bundled U-Net checkpoint, the older MSER proposal route, and manual curation tools to drastically accelerate segmentation of neural fibres (axons, dendrites, and potentially dendritic spines should the appropriate model be trained).

![FibreSight workbench with predicted ROI outlines and curation controls](docs/images/fibre-sight-workbench.png)

This repository comes with a `fibre_sight_ch2_v1.pt` checkpoint ([model card](MODEL_CARD.md); [methods and training record](METHODS.md)) trained on hand-labelled channel-2 axon images from our lab; the data are not included, but 3 example sessions (cropped) are bundled. The checkpoint's default operating point favours recall, with a confidence threshold of `0.25`, a minimum component size of `45` pixels, and four-view flip test-time augmentation.

## quick start

From the cloned repository root:

```powershell
conda env create -f environment.yml
conda activate fibre-sight
python -m fibre_sight
```

`run_fibre_sight.bat` starts the same workbench on Windows after the environment has been activated; `sh run_fibre_sight.sh` does the same on macOS and Linux. An editable pip installation is also sufficient:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
fibre-sight
```

Only the activation line differs on macOS and Linux:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
fibre-sight
```

Training and inference both default to automatic device selection: PyTorch uses CUDA when it is available, then Apple's MPS backend on Apple silicon, then CPU.

On macOS, use a native arm64 Python interpreter; `python -c "import platform; print(platform.machine())"` should print `arm64`. MPS acceleration also requires macOS 12.3 or later.

## quick demo

Three unlabelled 256 × 256 crops are included under `examples/`: two come from sessions in the training split, and `demo_test_ref_mat_ch2.npy` comes from one of the nine test sessions. The command below runs prediction on that test crop; the metrics in the [model card](MODEL_CARD.md) were calculated across the nine complete test sessions.

From the repository root:

```sh
fibre-sight-predict --image examples/demo_test_ref_mat_ch2.npy --out workspace/output/demo_predicted_ROI_dict.npy
```

The command uses the bundled checkpoint, chooses the available compute device, prints the number of predicted ROIs, and creates `workspace/output/` when needed. To inspect the same crop in the GUI, start `fibre-sight`, select `examples/demo_test_ref_mat_ch2.npy` under `channel-2 image`, open `predict`, then click `predict`.

## predict and curate

1. Select a two-dimensional channel-2 reference `.npy` file under `channel-2 image`.
2. Leave the bundled `trained model` selected, open `predict`, then click `predict`.
3. The confidence map is retained in memory, so `prediction threshold` and `minimum ROI area (pixels)` can be changed without another model pass. Click `rebuild ROIs` to replace the current proposals from that map.
4. Inspect the outlines, delete poor proposals, merge fragments when needed, and click `export ROIs`.

![FibreSight workflow overview across launch, prediction, labelling, and training](docs/images/workflow-overview.gif)

Higher `prediction threshold` values retain fewer, more confident ROIs. `minimum ROI area (pixels)` removes small connected components after thresholding. The `confidence` view shows the probability map beneath the ROI outlines; `ROI on / ROI off` hides those outlines without changing the current view. The [methods](METHODS.md#prediction-controls) record gives the defaults and the direction of every prediction and MSER control.

## training a new model

The bundled checkpoint covers only our lab's data. It has worked for multiple sensors and animals, but a different indicator, microscope or labelling convention may need its own model. The training loop is included below for retraining a model fit for new data.

### 1. label sessions

Select a session's `channel-2 image`, then open `label` to hand-label it. Delete, merge or fix proposals made by the MSER algorithm, then click `export ROIs`.

Arrange the results so each session directory holds both files:

```
workspace/labelled_sessions/<session>/processed_data/ref_mat_ch2.npy
workspace/labelled_sessions/<session>/processed_data/<name>_ROI_dict.npy
```

The reference array and the ROI dictionary must describe the same image.

### 2. build a manifest

```
fibre-sight-build-manifest
```

This scans the labelled sessions, writes `workspace/manifests/ch2_manifest.csv`, and prints the number of included sessions, the animals found, the total ROI count and the split sizes. Splits are assigned at session level with a 15% validation and 15% test fraction by default:

```
fibre-sight-build-manifest --val-fraction 0.2 --test-fraction 0.2 --seed 11
```

Use `--no-splits` to write the manifest without splits and assign an animal-held-out split manually in the CSV. `--source-root` and `--out` move the input and output locations.

### 3. edit a training recipe

Copy a shipped config and edit it without modifying the packaged file:

```
cp src/fibre_sight/configs/ch2_unet.yaml workspace/my_recipe.yaml
```

Paths inside the config resolve against the workspace root, so `data.manifest` of `manifests/ch2_manifest.csv` and `train.out_dir` of `output/runs` need no change at the default locations.

For a new dataset, adjust `data.patch_size` and `data.patches_per_image` to the field size, `data.foreground_fraction` to the mask density, and `data.normalise_percentiles` to the useful intensity range. `train.run_name` names the output directory; set `train.device` to `mps`, `cuda`, or `cpu` when automatic selection is unsuitable.

`ch2_unet_aux_recall.yaml` and `ch2_unet_512.yaml` are the two retained trials described in the [methods record](METHODS.md#retained-experiments).

### 4. train

```
fibre-sight-train --config workspace/my_recipe.yaml
```

Each epoch prints training loss, validation loss and validation Dice, and writes into `workspace/output/runs/<run_name>/`:

- `best.pt`, the weights with the highest mean validation patch Dice, measured at a probability threshold of `0.5` before any component filtering. This is the checkpoint to use for prediction.
- `latest.pt`, the final checkpoint.
- `history.csv`, rewritten every epoch, alongside a `history.png` under the figures directory.

Selection on validation Dice is fixed in the training code, so the saved epoch is often not the epoch where validation loss bottoms out; for the bundled model those were epoch 37 and epoch 59.

CUDA can use automatic mixed precision when `train.amp` is enabled; MPS training stays in float32, with pinned-memory loading disabled because it is CUDA-specific.

### 5. evaluate on the held-out split

```
fibre-sight-evaluate \
  --manifest workspace/manifests/ch2_manifest.csv \
  --checkpoint workspace/output/runs/<run_name>/best.pt \
  --split test \
  --tta \
  --out workspace/output/test_metrics.csv
```

This reports Dice, intersection over union, precision, recall and F2 per session. `--threshold` and `--min-size` override the postprocessing values for sweeping an operating point; unset, they fall back to the recipe defaults.

Choose the operating point after the weights are fixed, and inspect overlays alongside the numbers. Pixel scores describe foreground overlap and hide split and merge errors, so a setting that looks better on Dice can produce visibly worse ROIs. The [methods record](METHODS.md#choosing-the-released-operating-point) shows the sweep behind the bundled `0.25 / 45` default and why it sits between the recall-heavy and Dice-heavy optima.

### 6. use the new checkpoint

```
fibre-sight-predict \
  --image path/to/ref_mat_ch2.npy \
  --checkpoint workspace/output/runs/<run_name>/best.pt \
  --threshold 0.25 \
  --min-size 45 \
  --out workspace/output/predicted_ROI_dict.npy
```

Add `--no-tta` to disable four-view flip averaging, which is on by default. In the GUI, use `browse` beside the global `trained model` field to point at the new `best.pt` instead of the bundled checkpoint.

A new checkpoint that predicts well enough to seed curation replaces MSER as the proposal route for the next labelling round.

## common questions

### Will the bundled checkpoint work on images from another setup?

The checkpoint was trained on channel-2 references acquired and processed within one dLight dataset. Percentile normalisation absorbs simple changes in intensity scale, although a different indicator, microscope, field size, background texture or axon morphology can still change what the network recognises. New image domains should be checked against hand labels and may require retraining.

## workspace

Working data stay outside the package. A clone uses `workspace/` at the repository root by default:

```text
workspace/
    labelled_sessions/
    manifests/
    output/
        figures/
        runs/
```

Installed copies outside a checkout use `~/fibre-sight`. Setting `FIBRE_SIGHT_WORKSPACE` overrides both locations; all relative paths in the supplied training recipes resolve against that workspace.

The manifest builder expects two files in each labelled session:

```text
<session>/processed_data/ref_mat_ch2.npy
<session>/processed_data/*_ROI_dict.npy
```

`scan labelled sessions` writes a CSV containing the image and ROI paths, summaries, inclusion state, and the train, validation, or test split.

## command-line tools

Installing FibreSight adds four command-line tools alongside the GUI:

```text
fibre-sight-build-manifest
fibre-sight-train
fibre-sight-predict
fibre-sight-evaluate
```

Each command accepts `--help`. The same tools can be run as modules, for example `python -m fibre_sight.predict_rois --help`.

The prediction command uses the bundled checkpoint and its saved operating point by default:

```sh
fibre-sight-predict --image examples/demo_test_ref_mat_ch2.npy --out workspace/output/demo_predicted_ROI_dict.npy
```

Three training recipes are retained under `src/fibre_sight/configs/`: `ch2_unet.yaml` is the default 256-pixel crop recipe, `ch2_unet_aux_recall.yaml` is the support-target recall trial, and `ch2_unet_512.yaml` is the wider 512-pixel crop trial. These are working experiments; no exhaustive model comparison was run.

## verification

Run the standalone tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## repository scope

This repository contains the active FibreSight workbench extracted from a larger analysis repository. The pretrained checkpoint contains model weights and inference metadata; full source images and hand-labelled ROI dictionaries remain private. The three arrays under `examples/` are the cropped, unlabelled references described above.

The source code, bundled checkpoint, and demonstration arrays are released under the [MIT License](LICENSE). Mononoki 1.006 by Matthias Tellen is bundled for the interface under the [SIL Open Font License 1.1](src/fibre_sight/assets/fonts/mononoki/LICENSE); its [font note](src/fibre_sight/assets/fonts/mononoki/README.md) records the upstream release.

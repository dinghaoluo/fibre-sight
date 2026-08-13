# FibreSight

With tens of sessions and hundreds of neurite ROIs, manual curation can take days of effort, and the resulting segmentation also varies between experimenters. FibreSight reads two-dimensional channel-2 reference arrays (`ref_mat_ch2.npy`) and predicts candidate ROIs; the workbench combines a bundled U-Net checkpoint, the older MSER proposal route, and manual curation tools to accelerate segmentation of neural fibres (axons, dendrites, and potentially dendritic spines should the appropriate model be trained).

![FibreSight workbench with predicted ROI outlines and curation controls](docs/images/fibre-sight-workbench.png)

This repository comes with a `fibre_sight_ch2_v1.pt` checkpoint ([model card](MODEL_CARD.md); [methods and training record](METHODS.md)) trained on 53 hand-labelled channel-2 two-photon images from our lab; the data are not included, but 3 example sessions (cropped) are bundled as demos. The checkpoint reaches 0.93 recall on nine held-out test sessions, and its default operating point favours over-proposal: a confidence threshold of `0.25`, a minimum component size of `45` pixels, and four-view flip test-time augmentation. 

In addition to the plug-and-play prediction pipeline, the training workflow and all loss experiments are included, so that for a different indicator, microscope, or labelling convention, one can label sessions with the built-in MSER and curation tools, build a manifest, and train a new checkpoint on one's own data; see [How to Train Your Model™](TRAINING.md) for a guide to labelling, manifest building, and training.

## quick start

From the cloned repository root:

```powershell
conda env create -f environment.yml
conda activate fibre-sight
python -m fibre_sight  # note that it is an underscore, not a hyphen
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

On macOS, FibreSight uses a native arm64 Python interpreter; `python -c "import platform; print(platform.machine())"` should print `arm64`. MPS acceleration requires macOS 12.3 or later.

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

## model performance

The released checkpoint reached its best validation Dice of `0.7822` at epoch 37 of 80. On nine held-out test sessions, the released operating point (`0.25` threshold, `45` minimum pixels, four-view TTA) gives the following foreground-pixel scores:

| metric | mean |
| --- | ---: |
| Dice | 0.7227 |
| recall | 0.9261 |
| precision | 0.5958 |
| F2 | 0.8312 |

The operating point favours recall over precision. This is because a false positive only costs us one click to remove, whilst a missed axon costs the full manual labour of adding it into the dict.

![Training and validation loss with validation Dice across 80 epochs](docs/images/training-history.png)

Full metrics, the threshold sweep, and the training record are kept in the [methods](METHODS.md). The plotted values are available as [`training-history.csv`](docs/data/training-history.csv).

## common questions

### Will the bundled checkpoint work on images from another setup?

The checkpoint was trained on channel-2 references acquired and processed within one dLight dataset. Percentile normalisation absorbs simple changes in intensity scale, although a different indicator, microscope, field size, background texture, or axon morphology can still change what the network recognises. New image domains should be checked against hand labels and may require retraining.

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

Training recipes are kept under `src/fibre_sight/configs/`. The [training guide](TRAINING.md) describes each recipe and how to write a new one.

## verification

Run the standalone tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## repository scope

This repository contains the active FibreSight workbench extracted from a larger analysis repository. The pretrained checkpoint contains model weights and inference metadata; full source images and hand-labelled ROI dictionaries remain private. The three arrays under `examples/` are the cropped, unlabelled references described above.

The source code, bundled checkpoint, and demonstration arrays are released under the [MIT License](LICENSE). Mononoki 1.006 by Matthias Tellen is bundled for the interface under the [SIL Open Font License 1.1](src/fibre_sight/assets/fonts/mononoki/LICENSE); its [font note](src/fibre_sight/assets/fonts/mononoki/README.md) records the upstream release.

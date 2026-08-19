# FibreSight

[![tests](https://github.com/dinghaoluo/fibre-sight/actions/workflows/tests.yml/badge.svg)](https://github.com/dinghaoluo/fibre-sight/actions/workflows/tests.yml)

FibreSight is an integrated workbench for neurite segmentation, equipped with a GUI and CLI support. It is plug-and-play, with the bundled checkpoint U-Net model able to predict axonal ROIs from images out of the box, but the workbench also includes the entire training pipeline for use on drastically different datasets, from MSER-assisted data labelling and curation, to U-Net training and evaluation. To use the checkpoint model directly, see the `quick start` section below. For training using your own dataset, see [How to Train Your Model™](TRAINING.md).

![FibreSight workbench with predicted ROI outlines and curation controls](docs/images/fibre-sight-workbench.png)

The checkpoint U-Net model was developed from 53 hand-labelled two-photon sessions containing 1,296 curated ROIs. It reached 0.926 foreground-pixel recall on nine session-held-out images from the same four animals and acquisition source; the released operating point deliberately favours over-proposal. See the [model card](MODEL_CARD.md) and [methods record](METHODS.md) for the full evaluation.

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

For MPS acceleration on Apple silicon, use a native arm64 Python interpreter; `python -c "import platform; print(platform.machine())"` should print `arm64`. MPS requires macOS 12.3 or later. Other macOS installations fall back to CPU.

## quick demo

Three unlabelled 256 × 256 crops are included under `benchmarking/sources/`: two come from sessions in the training split, and `lab-fibresight-demo-test.npy` comes from one of the nine test sessions. The command below runs prediction on that test crop; the metrics in the [model card](MODEL_CARD.md) were calculated across the nine complete test sessions.

From the repository root:

```sh
fibre-sight-predict --image benchmarking/sources/lab-fibresight-demo-test.npy --out workspace/output/demo_predicted_ROI_dict.npy
```

The command uses the bundled checkpoint, chooses the available compute device, prints the number of predicted ROIs, and creates `workspace/output/` when needed. To inspect the same crop in the GUI, start `fibre-sight`, select `benchmarking/sources/lab-fibresight-demo-test.npy` under `channel-2 image`, open `predict`, then click `predict`.

## predict and curate

1. Select a two-dimensional channel-2 reference `.npy` file under `channel-2 image`.
2. Leave the bundled `trained model` selected, open `predict`, then click `predict`.
3. The confidence map is retained in memory, so `prediction threshold` and `minimum ROI area (pixels)` can be changed without another model pass. Click `rebuild ROIs` to replace the current proposals from that map.
4. Inspect the outlines, delete poor proposals, merge fragments when needed, and click `export ROIs`.

![FibreSight workflow overview across launch, prediction, labelling, and training](docs/images/workflow-overview.gif)

Higher `prediction threshold` values retain fewer, more confident ROIs. `minimum ROI area (pixels)` removes small connected components after thresholding. The `confidence` view shows the probability map beneath the ROI outlines; `ROI on / ROI off` hides those outlines without changing the current view. The [methods](METHODS.md#prediction-controls) record gives the defaults and the direction of every prediction and MSER control.

Canvas and ROI-table selections remain linked. The workbench supports multiselection, ROI import and export, deletion, merging, undo, pan and zoom, adjustable display black and white points, dark and light modes, and adjustable interface text size. In the `Label` tab, fixed proposals survive another MSER pass; the `Train` tab scans labelled sessions, previews the dataset, starts or stops training, scores a checkpoint, and saves diagnostic figures.

## model performance

The released checkpoint reached its best validation Dice of `0.7822` at epoch 37 of 80. On nine session-held-out test sessions from the same four animals and acquisition source, the released operating point (`0.25` threshold, `45` minimum pixels, four-view TTA) gives the following foreground-pixel scores:

| metric | mean |
| --- | ---: |
| Dice | 0.7227 |
| recall | 0.9261 |
| precision | 0.5958 |
| F2 | 0.8312 |

The operating point favours recall over precision. A false-positive proposal can be selected and deleted in the workbench; a missed axon must currently be drawn outside the workbench and imported.

![Training and validation loss with validation Dice across 80 epochs](docs/images/training-history.png)

Full metrics, the threshold sweep, and the training record are kept in the [methods](METHODS.md). The plotted values are available as [`training-history.csv`](docs/data/training-history.csv).

## common questions

### Will the bundled checkpoint work on images from another setup?

The checkpoint was trained on channel-2 references acquired and processed within one dLight dataset. Percentile normalisation absorbs simple changes in intensity scale, although a different indicator, microscope, field size, background texture, or axon morphology can still change what the network recognises. New image domains should be checked against hand labels and may require retraining.

### Can FibreSight start from a TIFF movie?

Yes, through the Python API. `preprocess_recording(...)` indexes paired TIFF pages, builds the reference, registers both channels with one shared motion estimate, and writes the movies and QC into NWB. The GUI still begins with a precomputed two-dimensional `.npy` reference image and ends with an ROI dictionary; activity-trace extraction belongs to the next stage.

The current motion-correction comparisons and their pinned external environments are recorded in the [benchmark notes](BENCHMARK.md).

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

ROI dictionaries use a legacy pickled NumPy format. `scan labelled sessions` writes a CSV containing the image and ROI paths, summaries, inclusion state, and the train, validation, or test split. The scanner currently infers the animal or subject group from the part of the session folder name before the first hyphen; inspect the `animal` and `split` columns before training, particularly when using another experimenter's folder convention.

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
fibre-sight-predict --image benchmarking/sources/lab-fibresight-demo-test.npy --out workspace/output/demo_predicted_ROI_dict.npy
```

Training recipes are kept under `src/fibre_sight/configs/`. The [training guide](TRAINING.md) describes each recipe and how to write a new one.

## verification

Run the standalone tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## cite FibreSight

If FibreSight contributes to published work, please cite the software version used:

```text
Luo, D. (2026). FibreSight (Version 0.1.3) [Computer software]. https://github.com/dinghaoluo/fibre-sight
```

GitHub's `Cite this repository` menu generates other citation formats from [`CITATION.cff`](CITATION.cff).

## repository scope

This repository contains the active FibreSight workbench extracted from a larger analysis repository. The pretrained checkpoint contains model weights and inference metadata; full source images and hand-labelled ROI dictionaries remain private. The ten registration references under `benchmarking/sources/` are cropped, unlabelled arrays with same-stem PNGs and a separate provenance record.

The source code, bundled checkpoint, and demonstration arrays are released under the [MIT License](LICENSE). Mononoki 1.006 by Matthias Tellen is bundled for the interface under the [SIL Open Font License 1.1](src/fibre_sight/assets/fonts/mononoki/LICENSE); its [font note](src/fibre_sight/assets/fonts/mononoki/README.md) records the upstream release.
